#!/bin/bash
# This script builds and runs the Docker container with GPU support

# Build the Docker image
docker build -t rocket_lander .

# Run the container with only CPU, and port forwarding for TensorBoard
docker run -it -p 6006:6006 \
  --env DISPLAY=$DISPLAY  \
  --env QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/rahul/sources/jhu/lbc_s25_project:/app \
  -w /app \
  lbcproject


# If you want to run training directly:
# docker run --gpus all -it -p 6006:6006 -v "$(pwd)":/app rocket_lander python simple_train.py
