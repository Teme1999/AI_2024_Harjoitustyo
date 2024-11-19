import numpy as np
import gymnasium as gym
from keras import models
import time
import os
import pygame

def run_agent():
    # Load the trained model
    model_path = 'dqn_mountain_car.keras'
    
    if os.path.exists(model_path):
        model = models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Trained model not found.")
        return

    # Create the environment
    env = gym.make('MountainCar-v0', render_mode='human')

    # Initialize pygame for event handling
    pygame.init()
    clock = pygame.time.Clock()

    state, _ = env.reset()
    state = np.reshape(state, [1, state.shape[0]])

    done = False
    total_reward = 0
    print(f"\nRunning agent continuously until failure...")

    while not done:
        env.render()

        # Get action from the trained model
        act_values = model.predict(state, verbose=0)
        action = np.argmax(act_values[0])

        next_state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, next_state.shape[0]])
        state = next_state

        # Close the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Closing the game window...")
                env.close()
                pygame.quit()
                return

        # Add a small delay to slow down the rendering for better visualization
        time.sleep(0.02)
    
    print(f"Completed at step: {abs(total_reward)}")
    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_agent()
