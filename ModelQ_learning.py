import optuna
from stable_baselines3 import PPO
import os
from SnakeEnvICM import CustomEnv  # Ensure this module is correctly imported
import time
import torch
import random
import numpy as np

# Function to evaluate the model's performance
def evaluate_model(env, model, num_episodes=10):
    total_rewards = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if isinstance(obs, tuple):
                obs = obs[0]
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_rewards += reward
            done = done or truncated
    return total_rewards / num_episodes

# Optuna's objective function
def objective(trial):
    current_time = time.time()
    models_dir = f"models/{int(current_time)}/"
    logdir = f"logs/{int(current_time)}/"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Clear environment before creating a new one
    env = CustomEnv()
    env.seed(seed)
    env.reset()

    # Hyperparameter suggestions
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical('n_steps', [2048, 4096, 8192])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    n_epochs = trial.suggest_int('n_epochs', 3, 30)
    gamma = trial.suggest_float('gamma', 0.90, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.80, 0.95)

    # Create and configure the PPO model
    model = PPO('MlpPolicy', env, verbose=0, tensorboard_log=logdir,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                device="cuda")

    TIMESTEPS_PER_ITER = 10000  # Reduced number of timesteps per iteration
    NUM_ITER = 3  # Number of training iterations

    try:
        for i in range(NUM_ITER):
            model.learn(total_timesteps=TIMESTEPS_PER_ITER, reset_num_timesteps=False)
            # Save the model after each training iteration
            model.save(os.path.join(models_dir, f'model_iteration_{i}.zip'))

        average_reward = evaluate_model(env, model)

        # Log results
        trial.set_user_attr("average_reward", average_reward)
        print(f"Trial {trial.number} finished with value: {average_reward} and parameters: {trial.params}.")
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        return float('inf')

    # Close the environment
    env.close()

    return average_reward

# Create and optimize the study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best trial
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
