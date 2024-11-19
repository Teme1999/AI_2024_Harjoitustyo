import numpy as np
import matplotlib.pyplot as plt

def visualize_rewards():
    # Load rewards per episode from the saved file
    rewards_per_episode = np.load('dqn_mountain_car.keras_rewards_per_episode.npy')

    EPISODES = len(rewards_per_episode)

    # Plot the total rewards per episode
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, EPISODES + 1), rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Agent Performance over Episodes during Training')
    plt.grid(True)
    plt.savefig('mountai_car_agent_performance_training.png')
    print("Plot saved as 'mountai_car_agent_performance_training.png'")

visualize_rewards()