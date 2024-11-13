import gymnasium as gym
import numpy as np
import os
from dqn_agent import DQNAgent

# TODO make a good reward function
def run_episode(agent, env, rewards_file, episode_idx):
    state_size = env.observation_space.shape[0]
    state, _ = env.reset()
    state = np.reshape(state, [1, state_size])
    
    # Track the starting position
    starting_position = state[0][0]
    furthest_distance = 0  # Initial furthest distance is zero

    goal_position = 0.5  # Goal position to reach
    goal_reward = 100  # Reward for reaching the goal

    for time_t in range(200):  # Max number of timesteps in each episode
        action = agent.act(state)  # Select an action based on the current state
        next_state, _, done, truncated, _ = env.step(action)  # Take the action

        current_position = next_state[0]

        # Calculate the distance from the starting position
        distance_from_start = abs(current_position - starting_position)

        # Update the furthest distance if this is the new maximum
        if distance_from_start > furthest_distance:
            furthest_distance = distance_from_start

        # Reward based on the furthest distance achieved in this episode
        reward = furthest_distance * 100  # Reward is the maximum distance from start

        # Large reward for reaching or surpassing the goal position
        if current_position >= goal_position:
            reward += goal_reward
            done = True  # End the episode once goal is reached
        
        state = np.reshape(next_state, [1, state_size])  # Update the state to the next state
        agent.remember(state, action, reward, next_state, done)  # Save experience to memory

        # Exit early if done or truncated
        if truncated:
            print(f"Episode: {episode_idx + 1}, score: {time_t}, total reward: {reward}")
            break

    # Replay memory and training outside the loop
    if len(agent.memory) > 32:  # Check if memory is sufficient for replay
        agent.replay(32)
    
    # Return the total reward at the end of the episode
    return reward

# Function to train DQN using sequential episodes
def train_dqn_mountain_car():
    # Create environment with render_mode='human' to display the game window
    env = gym.make("MountainCar-v0", render_mode="human")
    state_size = env.observation_space.shape[0]  # 2-dimensional state space (position, velocity)
    action_size = env.action_space.n  # 3 discrete actions: left, no action, right
    model_path = "dqn_mountain_car.keras"
    
    rewards_file = model_path + "_rewards_per_episode.npy"
    
    # Determine if we should continue training from a checkpoint
    continue_training = os.path.exists(model_path) and os.path.exists(rewards_file)
    
    # Initialize the DQNAgent with dynamic model path
    agent = DQNAgent(state_size, action_size, model_path, load_model=continue_training)

    # Load rewards and determine starting episode if continuing training
    if continue_training:
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
    train_dqn_mountain_car()
