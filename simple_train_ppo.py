#!/usr/bin/env python3
"""
Simple training script for the RocketLander environment using PPO from Stable Baselines 3.
"""

import os
import time
import gymnasium as gym
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from RocketLander import RocketLander

# Function to create a single environment
def make_env(rank, seed=0):
    def _init():
        env = RocketLander(render_mode=None)  # No rendering during training
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    # Create directories for logs and models
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"logs/ppo-rocket-{timestamp}"
    models_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)

    raw_env = RocketLander(render_mode=None)
    monitored = Monitor(raw_env)  

    # Create vectorized environments for parallel training
    n_envs = 4  # Number of parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=40000 // n_envs,  # Save every 40K steps (divided by number of environments)
        save_path=models_dir,
        name_prefix="ppo_rocket_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Create and train the PPO model
    model = PPO(
        policy="MlpPolicy",
        env=monitored,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=log_dir
    )

    # Train the model
    total_timesteps = 500000
    print(f"Training PPO model for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps, 
        callback=checkpoint_callback
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save the final model
    final_model_path = os.path.join(models_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()