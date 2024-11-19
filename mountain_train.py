import gymnasium as gym
import numpy as np
import os
from dqn_agent import DQNAgent

# Function to run a single episode
def run_episode(agent, env, train=True, batch_size=64, max_steps=200):
    total_reward = 0
    state, _ = env.reset()  # Reset environment and get the initial state
    state = state.reshape(1, -1)  # Ensure state is in the right shape
    terminated = False
    steps = 0
    highest_peak = state[0, 0]  # Track the highest position reached in the episode

    while not terminated and steps < max_steps:
        # Select an action using the DQN agent
        action = agent.act(state)

        # Perform the action in the environment
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_state = new_state.reshape(1, -1)

        # Track the highest peak reached
        if new_state[0, 0] > highest_peak:
            highest_peak = new_state[0, 0]

        # Check if the car has reached the goal
        if new_state[0, 0] >= 0.5 and not terminated:
            reward += 100  # Add 100 reward for reaching the goal
            terminated = True  # Terminate the episode if goal is reached

        # Remember experience for replay
        agent.remember(state, action, reward, new_state, terminated or truncated)

        # Update state
        state = new_state
        total_reward += reward
        steps += 1

        # Train the agent at every step
        if train and len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Add the highest peak reward once at the end of the episode, multiplied by 100
    total_reward += highest_peak * 100

    return total_reward

# Function to train the DQN on MountainCar-v0
def train_dqn_mountain_car(episodes, render=False):
    model_path = "dqn_mountain_car.keras"
    rewards_file = model_path + "_rewards_per_episode.npy"
    continue_training = os.path.exists(model_path) and os.path.exists(rewards_file)

    if continue_training:
        rewards_per_episode = np.load(rewards_file).tolist()
        start_episode = len(rewards_per_episode)
        print(f"Continuing training from episode {start_episode}")
    else:
        rewards_per_episode = []
        start_episode = 0

    # Initialize the MountainCar-v0 environment
    env = gym.make("MountainCar-v0", render_mode="human" if render else None)
    state_size = env.observation_space.shape[0]  # 2: position and velocity
    action_size = env.action_space.n  # 3 discrete actions: left, no acceleration, right

    # Initialize the DQNAgent
    agent = DQNAgent(state_size, action_size, model_path, load_model=continue_training)
    
    # Set agent's epsilon to a lower value to reduce exploration
    agent.epsilon = 0.01

    for e in range(start_episode, episodes):
        total_reward = run_episode(agent, env, train=True, batch_size=64, max_steps=200)
        rewards_per_episode.append(total_reward)

        print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}")

        # Save the model and rewards periodically
        if (e + 1) % 5 == 0:
            agent.model.save(model_path)
            np.save(rewards_file, np.array(rewards_per_episode))
            print(f"Checkpoint saved at episode {e + 1}")

    # Save the final model and rewards
    agent.model.save(model_path)
    np.save(rewards_file, np.array(rewards_per_episode))
    print("Training completed. Model and rewards saved.")
    env.close()

if __name__ == "__main__":
    train_dqn_mountain_car(500, render=True)
