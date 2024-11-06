import gymnasium as gym
import numpy as np
import os
from dqn_agent import DQNAgent

# Training function
def train_dqn(continue_training=False):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model_path = "dqn_cartpole.keras"
    agent = DQNAgent(state_size, action_size, model_path, load_model=continue_training)
    batch_size = 32
    EPISODES = 500

    # Load rewards and determine starting episode if continuing training
    if continue_training and os.path.exists('rewards_per_episode.npy'):
        rewards_per_episode = np.load('rewards_per_episode.npy').tolist()
        start_episode = len(rewards_per_episode)
        print(f"Continuing training from episode {start_episode + 1}")
    else:
        rewards_per_episode = []
        start_episode = 0

    # Early stopping parameters
    max_score = 500  # Maximum possible score in CartPole-v1
    patience = 5  # Number of consecutive episodes to achieve max_score before stopping
    consecutive_max_scores = 0

    try:
        for e in range(start_episode, EPISODES):
            state, _ = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            for time_t in range(500):
                action = agent.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                reward += 0.1  # Positive reward for each step the agent keeps the pole balanced
                reward = reward if not done else -1  # Adjusted penalty for losing an episode
                total_reward += reward
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done or truncated:
                    print(f"Episode: {e + 1}/{EPISODES}, score: {time_t}, total reward: {total_reward}, e: {agent.epsilon:.2f}")
                    rewards_per_episode.append(total_reward)
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            else:
                rewards_per_episode.append(total_reward)
                print(f"Episode: {e + 1}/{EPISODES}, score: {time_t}, total reward: {total_reward}, e: {agent.epsilon:.2f}")

            # Early stopping check
            if total_reward >= max_score:
                consecutive_max_scores += 1
                if consecutive_max_scores >= patience:
                    print(f"Early stopping triggered after {e + 1} episodes.")
                    agent.model.save(agent.model_path)
                    np.save('rewards_per_episode.npy', np.array(rewards_per_episode))
                    print("Model and rewards saved.")
                    break
            else:
                consecutive_max_scores = 0

            # Save checkpoint every 5 episodes
            if (e + 1) % 5 == 0:
                agent.model.save(agent.model_path)
                np.save('rewards_per_episode.npy', np.array(rewards_per_episode))
                print(f"Checkpoint saved at episode {e + 1}")
                print("Rewards per episode saved to 'rewards_per_episode.npy'")

    except KeyboardInterrupt:
        print('Training interrupted by user.')

    finally:
        # Save the model and rewards before exiting
        agent.model.save(agent.model_path)
        np.save('rewards_per_episode.npy', np.array(rewards_per_episode))
        print('Model and rewards saved.')

if __name__ == "__main__":
    continue_training = False  # Set to True to load from checkpoint and continue training
    train_dqn(continue_training=continue_training)
