import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from collections import deque
import matplotlib.pyplot as plt
import os
import pickle

# Check for GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version: ", tf.__version__)

# Create the environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # Should be 4
action_size = env.action_space.n  # Should be 2

# Define the DQNAgent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.95           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01     # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = deque(maxlen=2000)

        # Build the neural network model
        self.model = self._build_model()

    def _build_model(self):
        # Neural network architecture
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Return action with highest Q-value

    def replay(self):
        # Train the model using experiences sampled from memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in minibatch:
            target = reward  # If done, only reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target  # Update the Q-value for the action
            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train the network
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decrease exploration rate

    def save(self, name):
        # Save model weights and other parameters
        self.model.save_weights(name + '_model.h5')
        with open(name + '_params.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.epsilon, 'memory': self.memory}, f)

    def load(self, name):
        # Load model weights and other parameters
        self.model.load_weights(name + '_model.h5')
        with open(name + '_params.pkl', 'rb') as f:
            params = pickle.load(f)
            self.epsilon = params['epsilon']
            self.memory = params['memory']

agent = DQNAgent(state_size, action_size)

# Training parameters
episodes = 1000
output_dir = 'model_output/cartpole_dqn/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# To store rewards for visualization
rewards_list = []

# Load previous training state if available
resume_training = True  # Set to True if you want to resume training
checkpoint_name = output_dir + 'checkpoint'

if resume_training and os.path.isfile(checkpoint_name + '_model.h5'):
    agent.load(checkpoint_name)
    with open(output_dir + 'rewards_list.pkl', 'rb') as f:
        rewards_list = pickle.load(f)
    starting_episode = len(rewards_list)
    print(f"Resuming training from episode {starting_episode}")
else:
    starting_episode = 0

# Training loop
for e in range(starting_episode, episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        reward = reward if not done else -10  # Penalize if episode ends
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
            break
    rewards_list.append(total_reward)
    if len(agent.memory) > agent.batch_size:
        agent.replay()
    # Save the model and training state every 50 episodes
    if (e + 1) % 50 == 0:
        agent.save(output_dir + f"weights_{e+1}")
        # Also save the rewards list
        with open(output_dir + 'rewards_list.pkl', 'wb') as f:
            pickle.dump(rewards_list, f)
        # Save a checkpoint to resume training later
        agent.save(checkpoint_name)
        print("Checkpoint saved.")

# Plot the total rewards per episode
plt.figure(figsize=(12, 6))
plt.plot(rewards_list)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()

# Demonstrate the trained agent
for e in range(5):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):
        env.render()  # Visualize the environment
        action = np.argmax(agent.model.predict(state)[0])
        next_state, reward, done, _ = env.step(action)
        state = np.reshape(next_state, [1, state_size])
        if done:
            print(f"Test Episode: {e+1}/5, Score: {time}")
            break
env.close()