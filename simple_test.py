#!/usr/bin/env python3
"""
test_rocketlander_policy.py

Run a trained SB3 PPO policy on the RocketLander env for N episodes,
optionally rendering, and save per‐episode metrics to CSV.
"""

import argparse
import csv
import math
from RocketLander import RocketLander, FPS
from stable_baselines3 import PPO


def compute_distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def run_episodes(model_path: str, num_episodes: int, render: bool):
    # Load the trained policy
    model = PPO.load(model_path)

    results = []
    for ep in range(1, num_episodes + 1):
        # Instantiate env with or without rendering
        env = RocketLander(render_mode="human" if render else None)
        obs, _ = env.reset(seed=ep)

        # Precompute pad center for distance metrics
        pad_x = (env.helipad_x1 + env.helipad_x2) / 2
        pad_y = env.shipheight

        total_reward = 0.0
        steps = 0
        min_dist = float("inf")
        done = False
        truncated = False

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)

            # accumulate
            total_reward += reward
            steps += 1

            # track closest approach
            pos = env.lander.position  # b2Vec2
            dist = compute_distance((pos.x, pos.y), (pad_x, pad_y))
            min_dist = min(min_dist, dist)

            if render:
                env.render()

        # at episode end
        final_pos = env.lander.position
        final_dist = compute_distance((final_pos.x, final_pos.y), (pad_x, pad_y))
        
        landed = getattr(env, "_landed_ticks", 0) >= FPS
        crashed = getattr(env, "_game_over", False)
        success = landed and not crashed

        final_x, final_y = env.lander.position
        final_dist = compute_distance((final_x,final_y),(pad_x,pad_y))

        status = "SUCCESS" if success else "FAILURE"
        print(f"Episode {ep:2d}: {status}  "
              f"Reward={total_reward:.2f}  "
              f"Steps={steps}  "
              f"MinDist={min_dist:.3f}  "
              f"FinalDist={final_dist:.3f}")

        results.append({
            "episode":      ep,
            "success":      int(success),
            "total_reward": total_reward,
            "steps":        steps,
            "min_distance": min_dist,
            "final_distance": final_dist,
        })

        env.close()

    return results


def save_to_csv(results, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run a PPO policy on RocketLander for multiple episodes"
    )
    p.add_argument(
        "--model-path", type=str, required=True,
        help="Path to the .zip file of your trained PPO model"
    )
    p.add_argument(
        "--episodes", type=int, default=5,
        help="Number of episodes to run"
    )
    p.add_argument(
        "--render", action="store_true",
        help="If set, will render each step in a Pygame window"
    )
    p.add_argument(
        "--output", type=str, default="rocket_metrics.csv",
        help="CSV file to write per-episode metrics"
    )
    args = p.parse_args()

    metrics = run_episodes(args.model_path, args.episodes, args.render)
    save_to_csv(metrics, args.output)
    print(f"Saved metrics for {len(metrics)} episodes to {args.output}")
