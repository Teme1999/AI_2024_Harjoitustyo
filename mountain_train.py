import gymnasium as gym
import numpy as np
import os
from dqn_agent import DQNAgent
import time

# Training function for each episode (run sequentially)
def run_episode(agent, env, rewards_file, episode_idx):
    state_size = env.observation_space.shape[0]
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    prev_position = state[0][0]  # Initial position of the car (used to check progress)

    for time_t in range(200):  # Max number of timesteps in each episode
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)

        # Reward logic: base reward is 0, and agent gets +1 for reaching a new height
        if next_state[0] > prev_position:  # If the agent reached a new height
            reward = 1  # Reward for progress
        else:
            reward = 0  # No reward for staying at the same height or moving backward

        total_reward += reward
        prev_position = next_state[0]  # Update the previous position to the current one

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done or truncated:
            print(f"Episode: {episode_idx + 1}, score: {time_t}, total reward: {total_reward}")
            return total_reward  # Return the reward for this episode
        if len(agent.memory) > 32:  # Replay memory size
            agent.replay(32)

# Function to train DQN using sequential episodes
def train_dqn_mountain_car(continue_training=False):
    # Create environment with render_mode='human' to display the game window
    env = gym.make("MountainCar-v0", render_mode="human")
    state_size = env.observation_space.shape[0]  # 2-dimensional state space (position, velocity)
    action_size = env.action_space.n  # 3 discrete actions: left, no action, right
    model_path = "dqn_mountain_car.keras"
    
    rewards_file = model_path + "_rewards_per_episode.npy"
    
    # Initialize the DQNAgent with dynamic model path
    agent = DQNAgent(state_size, action_size, model_path, load_model=continue_training)

    # Load rewards and determine starting episode if continuing training
    if continue_training and os.path.exists(rewards_file):
        rewards_per_episode = np.load(rewards_file).tolist()
        start_episode = len(rewards_per_episode)
        print(f"Continuing training from episode {start_episode + 1}")
    else:
        rewards_per_episode = []
        start_episode = 0

    try:
        EPISODES = 500  # Number of episodes to train
        
        for e in range(start_episode, EPISODES):
            total_reward = run_episode(agent, env, rewards_file, e)
            rewards_per_episode.append(total_reward)  # Store the total reward for this episode

            print(f"Episode: {e + 1}/{EPISODES}, total reward: {total_reward}")

            # Optionally, save the model and rewards every few episodes
            if (e + 1) % 5 == 0:
                agent.model.save(agent.model_path)
                np.save(rewards_file, np.array(rewards_per_episode))  # Save rewards dynamically
                print(f"Checkpoint saved at episode {e + 1}")
                print(f"Rewards per episode saved to '{rewards_file}'")

    except KeyboardInterrupt:
        print('Training interrupted by user.')

    finally:
        # Save the model and rewards before exiting
        agent.model.save(agent.model_path)
        np.save(rewards_file, np.array(rewards_per_episode))  # Save rewards dynamically
        print('Model and rewards saved.')

if __name__ == "__main__":
    continue_training = False  # Set to True to load from checkpoint and continue training
    train_dqn_mountain_car(continue_training=continue_training)
