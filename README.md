# Lunar Lander RL Project with Stable-Baselines3 and Gymnasium

This project uses Stable-Baselines3 and Gymnasium to train a reinforcement learning agent to land the Lunar Lander in the LunarLander-v3 environment.

## Getting Started

### Build the Docker Image

```bash
docker build -t lbcproject .
```

### Run the Docker Container

```bash
docker run -it --rm -v $(pwd):/app -w /app lbcproject
```

## Training the Lunar Lander agent with Stable-Baselines3

To train an agent using Stable-Baselines3's PPO implementation, run the following command inside the Docker container:

```bash
python main.py
```

- This will start a PPO agent on the LunarLander-v3 environment for 50,000 timesteps (see `main.py`).
- You can adjust the training parameters in `main.py` as needed.
- Results and logs will be saved in the `/app` directory (your project folder).

## Project Structure

- `main.py`: Example script to verify environment and train with Stable-Baselines3.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container setup.

## References
- [Gymnasium LunarLander-v3 Docs](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)