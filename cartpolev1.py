import gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import matplotlib.pyplot as plt

# Hyperparameters
state_size = 4  # CartPole has 4 state variables
action_size = 2  # Two actions: left and right
batch_size = 64
gamma = 0.95  # Discount factor for future rewards
epsilon = 1.0  # Exploration rate (initial)
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Exploration decay
learning_rate = 0.001
episodes = 500

# Experience replay memory size
memory_size = 2000

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Neural network for Deep Q-learning model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, input_dim=state_size, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model

# DQN agent
class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.target_model = build_model()  # Target model for more stable training
        self.memory = deque(maxlen=memory_size)
        self.epsilon = epsilon
        self.update_target_network()

    def update_target_network(self):
        """Update target model with weights from the main model."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(action_size)  # Random action (explore)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])  # Best action (exploit)

    def store(self, state, action, reward, next_state, done):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + gamma * np.amax(self.target_model.predict(next_state)[0])
            
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
        
        if self.epsilon > epsilon_min:
            self.epsilon *= epsilon_decay

# Train the DQN agent and render periodically
def train_dqn(agent, episodes):
    scores = []
    render_every = 50  # Render every 50 episodes to visualize progress
    
    for e in range(episodes):
        state = env.reset()
        
        # Ensure the state is a numpy array
        if isinstance(state, tuple):
            state = state[0]  # Some environments may return a tuple (observation, info)
        
        state = np.array(state)  # Convert state to a NumPy array if it's not already
        state = np.reshape(state, [1, state_size])  # Ensure correct shape
        
        score = 0
        max_steps = 500
        
        for t in range(max_steps):
            if e % render_every == 0:  # Render every 50 episodes
                env.render()

            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Combine done and truncated flags to see if the episode has ended
            done = done or truncated

            # Ensure next_state is a numpy array
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            next_state = np.array(next_state)  # Convert next_state to a NumPy array
            next_state = np.reshape(next_state, [1, state_size])
            
            # Reward shaping to encourage longer episode durations
            reward = reward if not done else -10

            agent.store(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                print(f"Episode {e+1}/{episodes}, Score: {score}")
                agent.update_target_network()
                break

            agent.replay()
        
        scores.append(score)
        
        # Visualization update every 10 episodes
        if e % 10 == 0:
            plot_scores(scores)
    
    return scores


# Function to plot training progress
def plot_scores(scores):
    plt.figure(figsize=(10,5))
    plt.plot(scores, label='Score per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Performance on CartPole-v1')
    plt.show()

# Main
if __name__ == "__main__":
    agent = DQNAgent()
    scores = train_dqn(agent, episodes)
    env.close()  # Close the environment when done