import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend if running in a headless environment
import gymnasium as gym
import tensorflow as tf
from keras import models
import time
import os

# Uncomment the following line if running in a headless environment
# matplotlib.use('Agg')  # Use 'Agg' backend if running in a headless environment

def visualize_rewards():
    # Load rewards per episode from the saved file
    rewards_per_episode = np.load('rewards_per_episode.npy')

    EPISODES = len(rewards_per_episode)

    # Plot the total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, EPISODES + 1), rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance over Episodes during Training')
    plt.grid(True)
    plt.savefig('agent_performance_training.png')
    print("Plot saved as 'agent_performance_training.png'")

def run_agent_episodes():
    # Load the trained model
    model_path = 'dqn_cartpole.keras'
    if not os.path.exists(model_path):
        model_path = 'dqn_cartpole_weights.h5f'  # Try the weights file
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Trained model not found. Please ensure the model is saved as 'dqn_cartpole.keras' or 'dqn_cartpole_weights.h5f'.")
        return

    # Create the environment
    env = gym.make('CartPole-v1', render_mode='human')  # Use 'human' render mode for visualization

    # Define epsilon values for each episode to simulate improvement
    epsilon_values = np.linspace(1.0, 0.0, num=5)  # From 1.0 to 0.0 over 5 episodes

    for idx, epsilon in enumerate(epsilon_values):
        state, _ = env.reset()
        state = np.reshape(state, [1, state.shape[0]])
        total_reward = 0
        done = False
        step = 0

        print(f"\nEpisode {idx+1} with epsilon={epsilon:.2f}")

        while not done:
            env.render()

            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()  # Explore
            else:
                act_values = model.predict(state, verbose=0)
                action = np.argmax(act_values[0])  # Exploit

            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, next_state.shape[0]])
            state = next_state
            step += 1

            if done or truncated:
                print(f"Episode {idx+1} ended with total reward: {total_reward}")
                break

            # Add a small delay to slow down the rendering
            time.sleep(0.02)

    env.close()

if __name__ == "__main__":
    visualize_rewards()
    run_agent_episodes()
