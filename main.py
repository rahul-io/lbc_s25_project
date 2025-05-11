import gymnasium as gym
from RocketLander import RocketLander  # your file

env = RocketLander(render_mode="human")
obs, info = env.reset(seed=0)

done, truncated = False, False
for _ in range(10000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if env.render_mode == "human":
        env.render()
    if done or truncated:
        obs, info = env.reset()

env.close()