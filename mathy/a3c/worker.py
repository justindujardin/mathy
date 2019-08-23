import os
import threading
from multiprocessing import Queue
import time

import gym
import numpy as np
import tensorflow as tf

from .actor_critic_model import ActorCriticModel
from .config import A3CArgs
from .replay_buffer import ReplayBuffer
from .util import record


class A3CWorker(threading.Thread):

    args: A3CArgs

    # <GLOBAL_VARS>
    global_episode = 0
    global_moving_average_reward = 0
    save_every_n_episodes = 25
    save_lock = threading.Lock()
    # </GLOBAL_VARS>

    def __init__(
        self,
        args: A3CArgs,
        action_size: int,
        global_model: ActorCriticModel,
        optimizer,
        result_queue: Queue,
        worker_idx: int,
        shared_layers=None,
    ):
        super(A3CWorker, self).__init__()
        self.args = args
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.shared_layers = shared_layers
        self.optimizer = optimizer
        self.worker_idx = worker_idx
        self.env = gym.make(self.args.env_name)
        # Set up logging.
        self.writer = tf.summary.create_file_writer(
            os.path.join(self.args.model_dir, f"worker_{worker_idx}")
        )
        self.local_model = ActorCriticModel(
            args=args, predictions=self.action_size, shared_layers=shared_layers
        )
        self.local_model.maybe_load(self.env.reset())
        self.ep_loss = 0.0
        print(f"[Worker {worker_idx}] using env: {self.args.env_name}")

    def run(self):
        replay_buffer = ReplayBuffer()
        while A3CWorker.global_episode < self.args.max_eps:
            self.run_episode(replay_buffer)
            # TODO: Make this a subprocess? Python threads won't scale up well to
            #       many cores, I think.
        self.result_queue.put(None)

    def run_episode(self, replay_buffer: ReplayBuffer):
        current_state = self.env.reset()
        replay_buffer.clear()
        state_mask = self.env.action_space.mask
        ep_reward = 0.0
        ep_steps = 0
        self.ep_loss = 0

        time_count = 0
        done = False
        while not done:
            logits, _, probs = self.local_model.call_masked(current_state, state_mask)

            # Select a random action from the distribution with the given probabilities
            action = np.random.choice(len(probs), p=probs)

            # Take an env step
            new_state, reward, done, _ = self.env.step(action)
            state_mask = self.env.action_space.mask
            ep_reward += reward
            replay_buffer.store(current_state, action, reward)

            # The global step is incremented when the optimizer is applied, so check
            # and print summary data here.
            summary_interval = 25
            with tf.summary.record_if(
                lambda: tf.math.equal(
                    self.global_model.global_step % summary_interval, 0
                )
            ):
                with self.writer.as_default():
                    for var in self.local_model.trainable_variables:
                        tf.summary.histogram(
                            f"worker_{self.worker_idx}/{var.name}",
                            var,
                            step=self.global_model.global_step,
                        )

            if time_count == self.args.update_freq or done:
                self.update_global_network(done, new_state, replay_buffer)
                time_count = 0
                if done:
                    self.finish_episode(ep_reward, ep_steps)

            ep_steps += 1
            time_count += 1
            current_state = new_state

            # Workers wait between each step so that it's possible
            # to run more workers than there are CPUs available.
            time.sleep(self.args.worker_wait)

    def update_global_network(self, done, new_state, replay_buffer: ReplayBuffer):
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            total_loss = self.compute_loss(
                done, new_state, replay_buffer, self.args.gamma
            )
        self.ep_loss += total_loss
        # Calculate local gradients
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        # Push local gradients to global model
        self.optimizer.apply_gradients(
            zip(grads, self.global_model.trainable_weights),
            global_step=self.global_model.global_step,
        )
        # Update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())
        replay_buffer.clear()

    def finish_episode(self, episode_reward, episode_steps):
        A3CWorker.global_moving_average_reward = record(
            A3CWorker.global_episode,
            episode_reward,
            self.worker_idx,
            A3CWorker.global_moving_average_reward,
            self.result_queue,
            self.ep_loss,
            episode_steps,
            self.args.env_name,
        )
        # Write episode stats to Tensorboard
        with self.writer.as_default():
            name = self.args.env_name
            tf.summary.scalar(
                f"episode_reward",
                data=episode_reward,
                step=self.global_model.global_step,
            )
            tf.summary.scalar(
                f"{name}/ep_reward",
                data=episode_reward,
                step=self.global_model.global_step,
            )
            tf.summary.scalar(
                f"{name}/ep_steps",
                data=episode_steps,
                step=self.global_model.global_step,
            )
            agent_state = self.env.state.agent
            p_text = f"{agent_state.history[0].raw} = {agent_state.history[-1].raw}"
            outcome = "SOLVED" if episode_reward > 0.0 else "FAILED"
            out_text = f"{outcome}: {p_text}"
            tf.summary.text(
                f"{name}/ep_problems", data=out_text, step=self.global_model.global_step
            )
            if self.worker_idx == 0:
                tf.summary.scalar(
                    f"mean_episode_reward",
                    data=A3CWorker.global_moving_average_reward,
                    step=self.global_model.global_step,
                )

        # We must use a lock to save our model and to print to prevent data races.
        if A3CWorker.global_episode % A3CWorker.save_every_n_episodes == 0:
            with A3CWorker.save_lock:
                # Do this inside the lock so other threads can't also acquire the
                # lock in the time between when it's released and assigned outside
                # of the if conditional.
                A3CWorker.global_episode += 1
                self.global_model.save()
                self.writer.flush()
        else:
            A3CWorker.global_episode += 1

    def compute_loss(self, done, new_state, replay_buffer: ReplayBuffer, gamma=0.99):
        if done:
            reward_sum = 0.0  # terminal
        else:
            # Predict the reward using the local network
            _, values = self.local_model(new_state)
            reward_sum = tf.squeeze(values).numpy()

        # Get discounted rewards
        discounted_rewards = []
        for reward in replay_buffer.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        inputs = replay_buffer.to_features()
        logits, values = self.local_model(inputs)
        logits = tf.reshape(logits, [len(replay_buffer.actions), -1])

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
            labels=replay_buffer.actions, logits=logits
        )
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean(input_tensor=(0.5 * value_loss + policy_loss))
        return total_loss
