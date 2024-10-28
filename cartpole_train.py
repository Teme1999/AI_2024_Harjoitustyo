import numpy as np
import random
from collections import deque
import gymnasium as gym
import tensorflow as tf
from keras import models, layers, optimizers
import os

class DQNAgent:
    def __init__(self, state_size, action_size, load_model=False, model_path='dqn_cartpole.keras'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        
        self.gamma = 0.90    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0005

        self.model_path = model_path
        
        if load_model and os.path.exists(self.model_path):
            self.model = models.load_model(self.model_path)
            print(f"Loaded model from {self.model_path}")
            self.epsilon = self.epsilon_min
        else:
            self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = models.Sequential()
        model.add(layers.Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
   
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action
   
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([e[0] for e in minibatch])
        actions = np.array([e[1] for e in minibatch])
        rewards = np.array([e[2] for e in minibatch])
        next_states = np.vstack([e[3] for e in minibatch])
        dones = np.array([e[4] for e in minibatch])

        # Predict Q(s_t, a) for current states and next states
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)

        # Update target values
        for i in range(len(minibatch)):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

        # Train the model in one go
        self.model.fit(states, target, epochs=1, verbose=0)

        # Update exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_dqn(continue_training=False):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, load_model=continue_training)
    done = False
    batch_size = 64
    EPISODES = 500  # Maximum number of episodes to train

    # Early stopping parameters
    max_score = 500  # Maximum possible score in CartPole-v1
    patience = 5     # Number of consecutive episodes to achieve max_score before stopping
    consecutive_max_scores = 0  # Counter for consecutive max_score episodes

    rewards_per_episode = []  # List to store total rewards per episode

    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0  # Initialize total reward for this episode
        for time_t in range(500):
            # env.render()  # Uncomment to render training
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            reward = reward if not done else -1
            total_reward += reward  # Accumulate reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or truncated:
                print(f"Episode: {e+1}/{EPISODES}, score: {time_t}, total reward: {total_reward}, e: {agent.epsilon:.2f}")
                rewards_per_episode.append(total_reward)  # Save total reward
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        else:
            # If the loop wasn't broken (i.e., the episode didn't end early)
            rewards_per_episode.append(total_reward)
            print(f"Episode: {e+1}/{EPISODES}, score: {time_t}, total reward: {total_reward}, e: {agent.epsilon:.2f}")
        
        # Early stopping check
        if total_reward >= max_score:
            consecutive_max_scores += 1
            if consecutive_max_scores >= patience:
                print(f"Early stopping triggered after {e+1} episodes.")
                agent.model.save(agent.model_path)
                print("Model saved.")
                break
        else:
            consecutive_max_scores = 0  # Reset counter if max score not achieved

        # Save checkpoint every 25 episodes
        if (e + 1) % 25 == 0:
            agent.model.save(agent.model_path)
            print(f"Checkpoint saved at episode {e+1}")

    # Save rewards per episode to a file
    np.save('rewards_per_episode.npy', rewards_per_episode)
    print("Rewards per episode saved to 'rewards_per_episode.npy'")

if __name__ == "__main__":
    continue_training = False  # Set to True to load from checkpoint and continue training
    train_dqn(continue_training=continue_training)