import os
import threading

import gym
import numpy as np
import tensorflow as tf
from gym.wrappers import FlattenDictWrapper

from ..agent.features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_NODE_COUNT,
)
from . import record
from .actor_critic_model import ActorCriticModel
from .replay_buffer import ReplayBuffer


class A3CWorker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = None
    save_lock = threading.Lock()

    def __init__(
        self,
        state_size,
        action_size,
        global_model,
        optimizer,
        result_queue,
        idx,
        game_name,
        save_dir="/tmp",
        args=None,
        shared_units=128,
        shared_layers=None,
    ):
        super(A3CWorker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.result_queue = result_queue
        self.global_model = global_model
        self.shared_layers = shared_layers
        self.shared_units = shared_units
        self.optimizer = optimizer
        self.args = args
        self.local_model = ActorCriticModel(
            units=self.shared_units,
            predictions=self.action_size,
            shared_layers=shared_layers,
        )
        self.worker_idx = idx
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.env = FlattenDictWrapper(
            self.env,
            dict_keys=[
                FEATURE_FWD_VECTORS,
                FEATURE_BWD_VECTORS,
                FEATURE_LAST_BWD_VECTORS,
                FEATURE_LAST_FWD_VECTORS,
                FEATURE_LAST_RULE,
                FEATURE_NODE_COUNT,
            ],
        )

        self.save_dir = save_dir
        self.ep_loss = 0.0

    def ensure_state(self, state):
        return np.reshape(state, [1, self.state_size])

    def run(self):
        replay_buffer = ReplayBuffer()
        while A3CWorker.global_episode < self.args.max_eps:
            self.run_episode(replay_buffer)
        self.result_queue.put(None)

    def run_episode(self, replay_buffer: ReplayBuffer):
        current_state = self.env.reset()
        print(current_state)
        replay_buffer.clear()
        ep_reward = 0.0
        ep_steps = 0
        self.ep_loss = 0

        time_count = 0
        done = False
        while not done:
            logits, _ = self.local_model(
                tf.convert_to_tensor(value=current_state, dtype=tf.float32)
            )
            probs = tf.nn.softmax(logits)
            action = np.random.choice(self.action_size, p=probs.numpy()[0])
            new_state, reward, done, _ = self.env.step(action)
            if done:
                reward = -1
            ep_reward += reward
            replay_buffer.store(current_state, action, reward)

            if time_count == self.args.update_freq or done:
                self.update_global_network(done, new_state, replay_buffer)
                time_count = 0
                if done:
                    self.finish_episode(ep_reward, ep_steps)

            ep_steps += 1
            time_count += 1
            current_state = new_state

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
        self.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))
        # Update local model with new weights
        self.local_model.set_weights(self.global_model.get_weights())

        replay_buffer.clear()

    def finish_episode(self, episode_reward, episode_steps):
        """Complete an episode and save the model that produced the output
        if it's a record high return."""
        A3CWorker.global_moving_average_reward = record(
            A3CWorker.global_episode,
            episode_reward,
            self.worker_idx,
            A3CWorker.global_moving_average_reward,
            self.result_queue,
            self.ep_loss,
            episode_steps,
        )
        # We must use a lock to save our model and to print to prevent data races.
        if A3CWorker.best_score is None or episode_reward > A3CWorker.best_score:
            with A3CWorker.save_lock:
                model_full_path = os.path.join(
                    self.save_dir, "model_{}.h5".format(self.game_name)
                )
                print(
                    f" -- wrote best model {model_full_path}, score: {episode_reward}"
                )
                self.global_model.save_weights(model_full_path)
                A3CWorker.best_score = episode_reward
        A3CWorker.global_episode += 1

    def compute_loss(self, done, new_state, replay_buffer, gamma=0.99):
        if done:
            reward_sum = 0.0  # terminal
        else:
            # Predict the reward using the local network
            _, values = self.local_model(
                tf.convert_to_tensor(value=new_state[None, :], dtype=tf.float32)
            )
            reward_sum = tf.squeeze(values).numpy()

        # Get discounted rewards
        discounted_rewards = []
        for reward in replay_buffer.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        logits, values = self.local_model(
            tf.convert_to_tensor(
                value=np.vstack(replay_buffer.states), dtype=tf.float32
            )
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
            labels=replay_buffer.actions, logits=logits
        )
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean(input_tensor=(0.5 * value_loss + policy_loss))
        return total_loss
