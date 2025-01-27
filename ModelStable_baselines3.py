from stable_baselines3 import PPO
import os
from SnakeEnvICM import CustomEnv
import time

# Define directories for saving models and logs
current_time = time.time()
models_dir = f"models/{int(current_time)}/"
logdir = f"logs/{int(current_time)}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Initialize the custom environment
env = CustomEnv()
env.reset()

# Define a linear learning rate schedule
def linear_schedule(initial_value):
    def func(progress):
        return initial_value * (1 - progress)
    return func

# Use the best hyperparameters from Trial 16
model = PPO('MlpPolicy', env,
            verbose=1,
            tensorboard_log=logdir,
            learning_rate=linear_schedule(0.00013170936750225867),
            n_steps=8192,
            batch_size=256,
            n_epochs=13,
            gamma=0.9886338175632554,
            gae_lambda=0.8259917257649291,
            device="cuda")

TIMESTEPS = 10000  # Number of timesteps per training iteration
iters = 0  # Counter for iterations
max_iters = 1000  # Maximum number of iterations
save_interval = 5  # Interval for saving the model

while iters < max_iters:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    if iters % save_interval == 0:
        model.save(f"{models_dir}/model_{TIMESTEPS * iters}")

    if iters % save_interval == 0:
        # Enable visualization for 10 episodes
        env.show_gui = True  # Turn on visualization
        for _ in range(10):
            obs, _ = env.reset()  # Retrieve only the observation
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                # Do not call env.render() directly as visualization is already enabled
        env.show_gui = False  # Turn off visualization after finishing the games
        env.close()  # Close the pygame window if such logic exists
