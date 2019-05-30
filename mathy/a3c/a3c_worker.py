import os
import threading
import gym
import numpy as np

import tensorflow as tf
from . import record
from .replay_buffer import ReplayBuffer
from .actor_critic_model import ActorCriticModel


class A3CWorker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(
        self,
        state_size,
        action_size,
        global_model,
        opt,
        result_queue,
        idx,
        game_name="CartPole-v0",
        save_dir="/tmp",
        args=None,
    ):
        super(A3CWorker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.opt = opt
        self.args = args
        self.local_model = ActorCriticModel(self.state_size, self.action_size)
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name).unwrapped
        self.save_dir = save_dir
        self.ep_loss = 0.0

    def run(self):
        total_step = 1
        mem = ReplayBuffer()
        while A3CWorker.global_episode < self.args.max_eps:
            current_state = self.env.reset()
            mem.clear()
            ep_reward = 0.0
            ep_steps = 0
            self.ep_loss = 0

            time_count = 0
            done = False
            while not done:
                logits, _ = self.local_model(
                    tf.convert_to_tensor(value=current_state[None, :], dtype=tf.float32)
                )
                probs = tf.nn.softmax(logits)

                action = np.random.choice(self.action_size, p=probs.numpy()[0])
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    reward = -1
                ep_reward += reward
                mem.store(current_state, action, reward)

                if time_count == self.args.update_freq or done:
                    # Calculate gradient wrt to local model. We do so by tracking the
                    # variables involved in computing the loss by using tf.GradientTape
                    with tf.GradientTape() as tape:
                        total_loss = self.compute_loss(
                            done, new_state, mem, self.args.gamma
                        )
                    self.ep_loss += total_loss
                    # Calculate local gradients
                    grads = tape.gradient(
                        total_loss, self.local_model.trainable_weights
                    )
                    # Push local gradients to global model
                    self.opt.apply_gradients(
                        zip(grads, self.global_model.trainable_weights)
                    )
                    # Update local model with new weights
                    self.local_model.set_weights(self.global_model.get_weights())

                    mem.clear()
                    time_count = 0

                    if done:  # done and print information
                        A3CWorker.global_moving_average_reward = record(
                            A3CWorker.global_episode,
                            ep_reward,
                            self.worker_idx,
                            A3CWorker.global_moving_average_reward,
                            self.result_queue,
                            self.ep_loss,
                            ep_steps,
                        )
                        # We must use a lock to save our model and to print to prevent data races.
                        if ep_reward > A3CWorker.best_score:
                            with A3CWorker.save_lock:
                                print(
                                    "Saving best model to {}, "
                                    "episode score: {}".format(self.save_dir, ep_reward)
                                )
                                self.global_model.save_weights(
                                    os.path.join(
                                        self.save_dir,
                                        "model_{}.h5".format(self.game_name),
                                    )
                                )
                                A3CWorker.best_score = ep_reward
                        A3CWorker.global_episode += 1
                ep_steps += 1

                time_count += 1
                current_state = new_state
                total_step += 1
        self.result_queue.put(None)

    def compute_loss(self, done, new_state, memory, gamma=0.99):
        if done:
            reward_sum = 0.0  # terminal
        else:
            reward_sum = self.local_model(
                tf.convert_to_tensor(value=new_state[None, :], dtype=tf.float32)
            )[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(value=np.vstack(memory.states), dtype=tf.float32)
        )
        # Get our advantages
        advantage = (
            tf.convert_to_tensor(
                value=np.array(discounted_rewards)[:, None], dtype=tf.float32
            )
            - values
        )
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss
        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=memory.actions, logits=logits
        )
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean(input_tensor=(0.5 * value_loss + policy_loss))
        return total_loss

