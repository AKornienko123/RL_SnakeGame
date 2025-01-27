from stable_baselines3 import PPO
import os
import time
from SnakeEnvICM import CustomEnv
import torch
import random
import numpy as np


# Best trial parameters from Optuna
best_params = [
    {'learning_rate': 0.00013170936750225867, 'n_steps': 8192, 'batch_size': 256, 'n_epochs': 13, 'gamma': 0.9886338175632554, 'gae_lambda': 0.8259917257649291},
    {'learning_rate': 7.203125902895761e-05, 'n_steps': 8192, 'batch_size': 256, 'n_epochs': 28, 'gamma': 0.954259565358584, 'gae_lambda': 0.82217270183689},
    {'learning_rate': 0.00014804859781928148, 'n_steps': 8192, 'batch_size': 256, 'n_epochs': 10, 'gamma': 0.9481157521426704, 'gae_lambda': 0.815768793496382},
    {'learning_rate': 3.5984497852130865e-05, 'n_steps': 8192, 'batch_size': 256, 'n_epochs': 16, 'gamma': 0.9309307149585154, 'gae_lambda': 0.8183349623949033},
    {'learning_rate': 0.000252021058573941, 'n_steps': 2048, 'batch_size': 256, 'n_epochs': 14, 'gamma': 0.9527689374296026, 'gae_lambda': 0.8253129793689115},
    {'learning_rate': 0.00220782216292428, 'n_steps': 2048, 'batch_size': 256, 'n_epochs': 8, 'gamma': 0.9157692615547546, 'gae_lambda': 0.8595503560898705}
]

def evaluate_model(env, model, num_episodes=10):
    total_rewards = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if isinstance(obs, tuple):
                obs = obs[0]
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            done = done or truncated
        total_rewards += episode_reward
        print(f"Episode reward: {episode_reward}")
    return total_rewards / num_episodes

def run_trial_with_params(params, trial_number):
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

    env = CustomEnv()
    env.seed(seed)
    env.reset()

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir,
                learning_rate=params['learning_rate'],
                n_steps=params['n_steps'],
                batch_size=params['batch_size'],
                n_epochs=params['n_epochs'],
                gamma=params['gamma'],
                gae_lambda=params['gae_lambda'],
                device="cuda")

    TIMESTEPS_PER_ITER = 10000
    NUM_ITER = 10
    total_timesteps = 0

    try:
        for i in range(NUM_ITER):
            model.learn(total_timesteps=TIMESTEPS_PER_ITER, reset_num_timesteps=False)
            total_timesteps += TIMESTEPS_PER_ITER
            # Save the model after each training iteration
            model.save(os.path.join(models_dir, f'model_trial_{trial_number}_iteration_{i}.zip'))

        average_reward = evaluate_model(env, model)
        print(f"Trial {trial_number} finished with value: {average_reward}, timesteps: {total_timesteps} and parameters: {params}.")
    except Exception as e:
        print(f"Trial {trial_number} failed with exception: {e}")

    # Close the environment
    env.close()

# Run multiple trials with the best parameters
for i, params in enumerate(best_params):
    run_trial_with_params(params, i)
