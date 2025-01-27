from stable_baselines3 import PPO
import os
from SnakeEnv import CustomEnv

# Define the path and filename of the model
model_path = 'E:/Users/kibor/Desktop/zxc/models/1715332985/model_3000000.zip'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please check the path and file name.")

# Initialize the environment
env = CustomEnv()

# Load the model
model = PPO.load(model_path, env=env)
print(f"Model loaded from {model_path}")

# Enable visualization for demonstration
env.show_gui = True

# Run the model for several episodes
num_episodes = 10
for episode in range(num_episodes):
    obs, _ = env.reset()  # Ensure env.reset() returns only the observation
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

# Disable visualization and close the environment
env.show_gui = False
env.close()

print("Evaluation complete!")
