import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
import cv2
import matplotlib.pyplot as plt

# Hyperparameters
MEMORY_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 1000000
TARGET_UPDATE_FREQUENCY = 1000
TRAINING_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 500
REPLAY_START_SIZE = 1000
LEARNING_RATE = 0.00025

# Preprocess frames to grayscale and resize
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
    return resized_frame / 255.0

# Deep Q-Network with Debugging Prints
class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        print("[DEBUG] Initializing DQN agent with action size:", action_size)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        print("[DEBUG] DQN agent initialized successfully.")

    def build_model(self):
        print("[DEBUG] Building Q-network model.")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
            tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), loss='mse')
        print("[DEBUG] Q-network model built and compiled.")
        return model

    def update_target_model(self):
        print("[DEBUG] Updating target model weights.")
        self.target_model.set_weights(self.model.get_weights())
        print("[DEBUG] Target model updated.")

    def remember(self, state, action, reward, next_state, done):
        print(f"[DEBUG] Storing experience: action={action}, reward={reward}, done={done}")
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            print(f"[DEBUG] Taking random action: {action} (epsilon={self.epsilon:.2f})")
            return action
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        action = np.argmax(q_values[0])
        print(f"[DEBUG] Taking action based on Q-values: {action} (epsilon={self.epsilon:.2f})")
        return action

    def train(self):
        if len(self.memory) < REPLAY_START_SIZE:
            print(f"[DEBUG] Replay memory size is too small ({len(self.memory)}/{REPLAY_START_SIZE}). Skipping training.")
            return

        # Sample a batch from memory
        print("[DEBUG] Sampling a batch for training.")
        batch = random.sample(self.memory, BATCH_SIZE)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += GAMMA * np.amax(self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])

            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            q_values[0][action] = target

            states.append(state)
            targets.append(q_values[0])

        print("[DEBUG] Training the Q-network on sampled batch.")
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=BATCH_SIZE)
        
        # Decay epsilon
        if self.epsilon > EPSILON_END:
            self.epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY

# Main Training Loop with Debugging Prints
def train_agent(env):
    agent = DQNAgent(action_size=env.action_space.n)
    scores = []

    for e in range(TRAINING_EPISODES):
        print(f"\n[DEBUG] Starting episode {e+1}/{TRAINING_EPISODES}")
        state = preprocess_frame(env.reset())
        state = np.stack([state] * 4, axis=2)  # Stack 4 frames

        total_reward = 0
        for time in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_frame, reward, done, _ = env.step(action)
            next_state = preprocess_frame(next_frame)
            next_state = np.stack([next_state] * 4, axis=2)  # Stack next state frames
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.train()

            if done:
                print(f"[DEBUG] Episode finished after {time+1} steps with total reward: {total_reward}")
                break

        # Update target model periodically
        if e % TARGET_UPDATE_FREQUENCY == 0:
            print(f"[DEBUG] Updating target network at episode {e+1}.")
            agent.update_target_model()

        scores.append(total_reward)
        print(f"[DEBUG] Episode {e+1} finished with total reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Save video/render every 100 episodes for graphical demonstration
        if e % 100 == 0:
            print(f"[DEBUG] Rendering game at episode {e+1}.")
            render_game(agent, env)

    # Plot learning curve
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# Render and display game with Debugging Prints
def render_game(agent, env):
    print("[DEBUG] Starting game rendering.")
    state = preprocess_frame(env.reset())
    state = np.stack([state] * 4, axis=2)

    for _ in range(MAX_STEPS_PER_EPISODE):
        action = agent.act(state)
        next_frame, _, done, _ = env.step(action)
        env.render()  # Display the game
        next_state = preprocess_frame(next_frame)
        next_state = np.stack([next_state] * 4, axis=2)
        state = next_state

        if done:
            print("[DEBUG] Game rendering finished.")
            break

# Main execution
if __name__ == "__main__":
    env = gym.make('SpaceInvaders-v4')  # Change to the desired environment
    train_agent(env)