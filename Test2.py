from SnakeEnv import CustomEnv
import random

env = CustomEnv()
episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()  # Start with the initial state
    while not done:  # Use the "done" flag to control the loop
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, terminated, truncated, info = env.step(random_action)
        print('reward', reward)
        if terminated:  # Check if the game is over
            print("Game Over. Restarting...")
            break  # Exit the loop to start a new episode
