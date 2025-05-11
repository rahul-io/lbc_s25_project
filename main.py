import gymnasium as gym
from stable_baselines3 import PPO
from RocketLander import RocketLander

if __name__ == "__main__":
    # Create the RocketLander environment
    env = RocketLander(enable_wind=False, render_mode="human")
    obs, info = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # Replace with your policy or agent
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
