# AI_2024
Project work for 2024 AI course


Requirements:

    Python 3.11.10
    TensorFlow 2.17.0
    Gymnasium 1.0.0 (not Gym!)


File Descriptions:

    dqn_agent.py:
    Contains the implementation of the DQN model, including the neural network architecture and the training process.

    cartpole_train.py:
    Script to train the DQN agent on the CartPole-v1 environment.

    cartpole_test.py:
    Script to test the trained CartPole-v1 agent.

    cartpole_plot_rewards.py:
    Generates a plot of the agent's training rewards per episode and saves it as 
    cartpole_agent_performance_training.png.

    dqn_cartpole.keras:
    Is our trained model for the CartPole-v1 game.

    dqn_cartpole.keras_rewards_per_episode.npy
    Is reward data per episode for the dqn_cartpole.keras model.

    mountaincar_train.py:
    Script to train the DQN agent on the MountainCar-v0 environment.

    mountaincar_test.py:
    Script to test the trained MountainCar-v0 agent.

    mountain_car_plot_rewards.py:
    Generates a plot of the agent's training rewards per episode and saves it as mountain_car_agent_performance_training.png.

    dqn_mountain_car.keras:
    Is our trained model for the MountainCar-v0 game.

    dqn_mountain_car.keras_rewards_per_episode.npy
    Is reward data per episode for the dqn_mountain_car.keras model.


Analysis:
    
    cartpole_train.py trains the model by giving 0.1 reward for each step it doesn't drop the cartpole
    and giving -1 for failure.
    dqn_cartpole.keras_rewards_per_episode.npy show that model made consistent improvement until 85 episode 
    then model got near 0 reward for 10 episodes. 
    After thoes episodes the model reached 500 reward for 5 episodes which ends the training. 
    carpole_test.py can be used and it shows the model playing cartpole-v1 flawlessly.
    
    mountain_car_train.py trains the model by giving reward depending how far the car reaches. 
    0.5 is maximum distance which rewards the ai with 70 for distance, 100 for reaching the goal,
    velocity*10 and -1 for each step it took to reach the goal. 
    Score of 0 is pretty fast taking aroud 100 steps to complete


Results:
    
    This project demonstrates the ability of a DQN agent to learn and adapt to diverse 
    environments like CartPole-v1 and MountainCar-v0. 
    Training rewards and testing performance highlight the effectiveness of the implemented model. 
    The dqn is able to learn both games with enough training and then the model can be used to complete the games continuously.


Quotation:

    The base for the code and comments were generated with the assistance of ChatGPT. 
    Modifications and logical reward systems and variable values were tested and added manually.
