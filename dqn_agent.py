import gymnasium as gym
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers, optimizers, losses
import os

# Define the DQN agent class
class DQNAgent:
    def __init__(self, state_size, action_size, load_model=False, model_path='dqn_cartpole.keras'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # Increased learning rate for faster convergence
        self.tau = 0.1  # Soft update parameter for target network
        self.model_path = model_path

        # Main and target networks
        self.model = self._build_model()
        self.target_model = self._build_model()

        if load_model and os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.target_model.set_weights(self.model.get_weights())
            print(f"Loaded model from {self.model_path}")
            self.epsilon = self.epsilon_min
        else:
            self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate), loss=losses.MeanSquaredError())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([e[0] for e in minibatch])
        targets = self.model.predict(states, verbose=0)
        next_states = np.vstack([e[3] for e in minibatch])
        target_next = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.amax(target_next[i])
            targets[i][action] = target

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Soft update target model
        self._update_target_model()

    def _update_target_model(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
