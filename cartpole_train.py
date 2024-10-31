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

# Training function
def train_dqn(continue_training=False):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, load_model=continue_training)
    batch_size = 32
    EPISODES = 500

    # Load rewards and determine starting episode if continuing training
    if continue_training and os.path.exists('rewards_per_episode.npy'):
        rewards_per_episode = np.load('rewards_per_episode.npy').tolist()
        start_episode = len(rewards_per_episode)
        print(f"Continuing training from episode {start_episode + 1}")
    else:
        rewards_per_episode = []
        start_episode = 0

    # Early stopping parameters
    max_score = 500  # Maximum possible score in CartPole-v1
    patience = 5  # Number of consecutive episodes to achieve max_score before stopping
    consecutive_max_scores = 0

    try:
        for e in range(start_episode, EPISODES):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            for time_t in range(500):
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                reward += 0.1  # Positive reward for each step the agent keeps the pole balanced
                reward = reward if not done else -1  # Adjusted penalty for losing an episode
                total_reward += reward
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done or truncated:
                    print(f"Episode: {e + 1}/{EPISODES}, score: {time_t}, total reward: {total_reward}, e: {agent.epsilon:.2f}")
                    rewards_per_episode.append(total_reward)
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            else:
                rewards_per_episode.append(total_reward)
                print(f"Episode: {e + 1}/{EPISODES}, score: {time_t}, total reward: {total_reward}, e: {agent.epsilon:.2f}")

            # Early stopping check
            if total_reward >= max_score:
                consecutive_max_scores += 1
                if consecutive_max_scores >= patience:
                    print(f"Early stopping triggered after {e + 1} episodes.")
                    agent.model.save(agent.model_path)
                    np.save('rewards_per_episode.npy', np.array(rewards_per_episode))
                    print("Model and rewards saved.")
                    break
            else:
                consecutive_max_scores = 0

            # Save checkpoint every 5 episodes
            if (e + 1) % 5 == 0:
                agent.model.save(agent.model_path)
                np.save('rewards_per_episode.npy', np.array(rewards_per_episode))
                print(f"Checkpoint saved at episode {e + 1}")
                print("Rewards per episode saved to 'rewards_per_episode.npy'")

    except KeyboardInterrupt:
        print('Training interrupted by user.')

    finally:
        # Save the model and rewards before exiting
        agent.model.save(agent.model_path)
        np.save('rewards_per_episode.npy', np.array(rewards_per_episode))
        print('Model and rewards saved.')

if __name__ == "__main__":
    continue_training = False  # Set to True to load from checkpoint and continue training
    train_dqn(continue_training=continue_training)