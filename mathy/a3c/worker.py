import os
import threading
from multiprocessing import Queue
import time

import gym
import numpy as np
import tensorflow as tf

from ..core.expressions import MathTypeKeysMax
from .actor_critic_model import ActorCriticModel
from .config import A3CArgs
from .replay_buffer import ReplayBuffer
from .util import record, cat_entropy, entropy_beta_for_training_step
from trfl import discrete_policy_entropy_loss, discrete_policy_gradient_loss


class A3CWorker(threading.Thread):

    args: A3CArgs

    # <GLOBAL_VARS>
    global_episode = 0
    global_moving_average_reward = 0
    save_every_n_episodes = 50
    request_quit = False
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
        writer: tf.summary.SummaryWriter,
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
        self.writer = writer
        self.local_model = ActorCriticModel(
            args=args,
            predictions=self.action_size,
            shared_layers=shared_layers,
            optimizer=self.optimizer,
        )
        self.local_model.maybe_load(self.env.reset())
        self.ep_loss = 0.0
        self.ep_pi_loss = 0.0
        self.ep_value_loss = 0.0
        self.ep_entropy_loss = 0.0

        print(f"[Worker {worker_idx}] using env: {self.args.env_name}")

    def run(self):
        replay_buffer = ReplayBuffer()
        while (
            A3CWorker.global_episode < self.args.max_eps
            and A3CWorker.request_quit is False
        ):
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
            # Select a random action from the distribution with the given probabilities
            # if np.random.random() < self.args.exploration_greedy_epsilon:
            #     action = self.env.action_space.sample()
            # else:
            _, _, probs = self.local_model.call_masked(current_state, state_mask)
            action = np.random.choice(len(probs), p=probs)

            # Take an env step
            new_state, reward, done, _ = self.env.step(action)
            state_mask = self.env.action_space.mask
            ep_reward += reward
            replay_buffer.store(current_state, action, reward)

            self.maybe_write_histograms()

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

    def maybe_write_episode_summaries(self, episode_reward: float, episode_steps: int):
        # Track metrics for all workers
        name = self.args.env_name
        step = self.global_model.global_step
        with self.writer.as_default():

            tf.summary.scalar(
                f"rewards/worker_{self.worker_idx}/episodes",
                data=episode_reward,
                step=step,
            )
            tf.summary.scalar(
                f"steps/worker_{self.worker_idx}/ep_steps",
                data=episode_steps,
                step=step,
            )

            # TODO: track per-worker averages and log them
            # tf.summary.scalar(
            #     f"rewards/worker_{self.worker_idx}/mean_episode_reward",
            #     data=episode_reward,
            #     step=step,
            # )

            agent_state = self.env.state.agent
            p_text = f"{agent_state.history[0].raw} = {agent_state.history[-1].raw}"
            outcome = "SOLVED" if episode_reward > 0.0 else "FAILED"
            out_text = f"{outcome}: {p_text}"
            tf.summary.text(
                f"{name}/worker_{self.worker_idx}/summary", data=out_text, step=step
            )

            if self.worker_idx == 0:
                # Track global model metrics
                tf.summary.scalar(
                    f"rewards/mean_episode_reward",
                    data=A3CWorker.global_moving_average_reward,
                    step=step,
                )
                # Track global model metrics
                tf.summary.scalar(
                    f"params/policy_entropy_beta",
                    data=entropy_beta_for_training_step(self.args, step.numpy()),
                    step=step,
                )

    def maybe_write_histograms(self):
        if self.worker_idx != 0:
            return
        # The global step is incremented when the optimizer is applied, so check
        # and print summary data here.
        summary_interval = 10
        with self.writer.as_default():
            with tf.summary.record_if(
                lambda: tf.math.equal(
                    self.global_model.global_step % summary_interval, 0
                )
            ):
                for var in self.local_model.trainable_variables:
                    tf.summary.histogram(
                        var.name, var, step=self.global_model.global_step
                    )

    def update_global_network(self, done, new_state, replay_buffer: ReplayBuffer):
        # Calculate gradient wrt to local model. We do so by tracking the
        # variables involved in computing the loss by using tf.GradientTape
        with tf.GradientTape() as tape:
            pi_loss, value_loss, entropy_loss, total_loss = self.compute_loss(
                done, new_state, replay_buffer, self.args.gamma
            )
        self.ep_loss += total_loss
        self.ep_pi_loss += pi_loss
        self.ep_value_loss += value_loss
        self.ep_entropy_loss += entropy_loss
        # Calculate local gradients
        grads = tape.gradient(total_loss, self.local_model.trainable_weights)
        # Push local gradients to global model
        self.optimizer.iterations = self.global_model.global_step
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
            self.ep_pi_loss,
            self.ep_value_loss,
            self.ep_entropy_loss,
            self.ep_loss,
            episode_steps,
            self.args.env_name,
        )
        self.maybe_write_episode_summaries(episode_reward, episode_steps)

        # We must use a lock to save our model and to print to prevent data races.
        if A3CWorker.global_episode % A3CWorker.save_every_n_episodes == 0:
            self.write_global_model()
        else:
            A3CWorker.global_episode += 1

    def write_global_model(self, increment_episode=True):
        with A3CWorker.save_lock:
            # Do this inside the lock so other threads can't also acquire the
            # lock in the time between when it's released and assigned outside
            # of the if conditional.
            if increment_episode is True:
                A3CWorker.global_episode += 1
                self.global_model.save()

    def compute_loss(self, done, new_state, replay_buffer: ReplayBuffer, gamma=0.99):
        step = self.global_model.global_step
        if done:
            reward_sum = 0.0  # terminal
        else:
            # Predict the reward using the local network
            _, values, _ = self.local_model(new_state)
            reward_sum = tf.squeeze(values).numpy()

        # Get discounted rewards
        discounted_rewards = []
        for reward in replay_buffer.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + gamma * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(
            value=np.array(discounted_rewards)[:, None], dtype=tf.float32
        )

        batch_size = len(replay_buffer.actions)

        inputs = replay_buffer.to_features()
        logits, values, masked = self.local_model(inputs, apply_mask=False)
        logits = tf.reshape(logits, [batch_size, -1])
        masked_flat = tf.reshape(masked, [batch_size, -1])

        # # reshape to the logits dimensions
        # action_labels = tf.convert_to_tensor(replay_buffer.actions, dtype=tf.int32)
        # reward_values = tf.convert_to_tensor(replay_buffer.rewards, dtype=tf.float32)

        # # TODO: An extra dimension is needed, so expand the dims.. this feels wrong?
        # logits = tf.expand_dims(logits, axis=1)
        # # logits = tf.expand_dims(masked, axis=1)
        # action_labels = tf.expand_dims(action_labels, axis=1)
        # reward_values = tf.expand_dims(reward_values, axis=1)

        # Calculate entropy and policy loss
        h_loss = discrete_policy_entropy_loss(logits, normalise=True)
        # pi_loss = discrete_policy_gradient_loss(logits, action_labels, reward_values)

        # Advantage is the difference between the final calculated discount
        # rewards, and the current Value function prediction of the rewards
        advantage = discounted_rewards - values

        # Value loss
        value_loss = advantage ** 2

        # Policy Loss

        # We calculate policy loss from the masked logits to keep
        # the error from exploding when irrelevant (masked) logits
        # have large values. Because we apply a mask for all operations
        # we don't care what those logits are, unless they're part of
        # the mask.
        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=replay_buffer.actions, logits=masked_flat
        )
        policy_loss *= tf.stop_gradient(advantage)

        value_loss *= 0.5

        total_loss = tf.reduce_mean(value_loss + policy_loss + h_loss.loss)
        with self.writer.as_default():
            tf.summary.scalar(
                f"losses/worker_{self.worker_idx}/loss", data=total_loss, step=step
            )
            tf.summary.scalar(
                f"losses/worker_{self.worker_idx}/policy_loss",
                data=tf.reduce_mean(policy_loss),
                step=step,
            )
            tf.summary.scalar(
                f"losses/worker_{self.worker_idx}/value_loss",
                data=tf.reduce_mean(value_loss),
                step=step,
            )
            tf.summary.scalar(
                f"values/worker_{self.worker_idx}/entropy_loss",
                data=tf.reduce_mean(h_loss.loss),
                step=step,
            )
            tf.summary.scalar(
                f"values/worker_{self.worker_idx}/advantage",
                data=tf.reduce_sum(advantage),
                step=step,
            )
            tf.summary.scalar(
                f"values/worker_{self.worker_idx}/entropy",
                data=tf.reduce_mean(h_loss.extra.entropy),
                step=step,
            )

        return (
            tf.reduce_mean(policy_loss),
            tf.reduce_mean(value_loss),
            tf.reduce_mean(h_loss.loss),
            total_loss,
        )
