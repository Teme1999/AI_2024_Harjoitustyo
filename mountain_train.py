import gymnasium as gym
import numpy as np
import os
from dqn_agent import DQNAgent
import time

# TODO make good reward fucntion
def run_episode(agent, env, rewards_file, episode_idx):
    state_size = env.observation_space.shape[0]
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    prev_position = state[0][0]  # Initial position of the car
    prev_velocity = state[0][1]  # Initial velocity of the car
    episode_done = False  # Flag to track episode completion
    
    # Define constants for goal completion and speed rewards
    goal_position = 0.5  # Set the goal position (where we want the agent to reach)
    goal_reward = 100  # Large reward for completing the episode
    progress_reward = 0.1  # Small reward for positive progress
    speed_bonus = 0.5  # Reward for moving fast in the right direction
    penalty_for_moving_back = -1.0  # Penalty for moving backward (negative velocity)

    for time_t in range(200):  # Max number of timesteps in each episode
        action = agent.act(state)  # Select an action based on the current state
        next_state, _, done, truncated, _ = env.step(action)  # Take the action

        current_position = next_state[0]
        current_velocity = next_state[1]

        # Reward for reaching a new height (positive progress)
        if current_position > prev_position:
            reward = progress_reward  # Small reward for upward progress
        else:  # Moving backward or staying in the same place
            reward = penalty_for_moving_back
        
        # Reward for increasing speed (the faster it moves, the better)
        if current_velocity > prev_velocity:  # If the agent's speed is increasing in the right direction
            reward += speed_bonus  # Additional reward for increasing speed
        
        # Large reward for reaching the goal position (0.5 or higher)
        if current_position >= goal_position:
            reward += goal_reward
            done = True  # End the episode once goal is reached
        
        total_reward += reward
        prev_position = current_position  # Update the previous position
        prev_velocity = current_velocity  # Update the previous velocity
        
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)  # Save experience to memory
        state = next_state
        
        # Exit early if done or truncated
        if done or truncated:
            print(f"Episode: {episode_idx + 1}, score: {time_t}, total reward: {total_reward}")
            episode_done = True
            break
        
    # Replay memory and training outside the loop
    if len(agent.memory) > 32:  # Check if memory is sufficient for replay
        agent.replay(32)
    
    # Return the total reward at the end of the episode
    return total_reward

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
