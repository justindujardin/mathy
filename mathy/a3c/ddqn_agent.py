# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import tensorflow as tf
from typing import List, Optional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


EPISODES = 5000


class DDQNAgent(tf.keras.Model):
    """DDQN agent from: https://github.com/keon/deep-q-learning"""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=3e-4,
        discount_rate=0.95,
        exploration_rate=1.0,
        exploration_rate_min=0.01,
        exploration_rate_decay=0.99,
        max_memory_size=2000,
        units: int = 128,
        shared_layers: Optional[List[tf.keras.layers.Layer]] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.units = units
        self.shared_layers = shared_layers
        self.memory = deque(maxlen=max_memory_size)
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (
            K.abs(error) - clip_delta
        )

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(self.units, input_dim=self.state_size, activation="relu"))
        if self.shared_layers is not None:
            for layer in self.shared_layers:
                model.add(layer)
        model.add(Dense(self.units, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.discount_rate * np.amax(t)
                # target[0][action] = reward + self.discount_rate * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_rate_min:
            self.exploration_rate *= self.exploration_rate_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


class DDQNModel(tf.keras.Model):
    def __init__(self, state_size, action_size, shared_layers=None):
        super(DDQNModel, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.shared_layers = shared_layers
        self.in_dense = tf.keras.layers.Dense(128)
        self.value_dense = tf.keras.layers.Dense(128)
        self.pi_logits = tf.keras.layers.Dense(action_size)
        self.value_logits = tf.keras.layers.Dense(1)

    def _build_control_policy(self):
        self.aux_agent = DDQNAgent(
            self.state_size,
            self.action_size,
            units=self.shared_units,
            shared_layers=self.shared_layers,
        )

    def _run_control_policy(self, state):
        # env.render()
        batch_size = 24
        state = self.ensure_state(state)
        action = self.aux_agent.act(state)
        next_state, reward, done, _ = self.aux_env.step(action)
        reward = reward if not done else -10
        next_state = self.ensure_state(next_state)
        self.aux_agent.remember(state, action, reward, next_state, done)
        state = next_state
        with A3CWorker.save_lock:
            if done:
                self.aux_agent.update_target_model()
            elif len(self.aux_agent.memory) > batch_size:
                self.aux_agent.replay(batch_size)
        return next_state, done

    def call(self, inputs):
        inputs = self.in_dense(inputs)
        if self.shared_layers is not None:
            for layer in self.shared_layers:
                inputs = layer(inputs)
        logits = self.pi_logits(inputs)
        values = self.value_logits(self.value_dense(inputs))
        return logits, values

