#!/bin/bash
# Run training with Isaac Lab

cd /home/ubuntu/Orphan_MVP1

# Set up Isaac Lab path
export PYTHONPATH="/home/ubuntu/IsaacLab/source:$PYTHONPATH"

# Run training
# Use --headless with --enable_cameras to avoid dependency issues with rendering mode
/opt/IsaacSim/python.sh scripts/train_ppo_isaaclab.py \
    --num-envs 4096 \
    --output-dir outputs \
    --train-config configs/train/DroneTankPPO.yaml \
    --headless \
    --enable_cameras \
    "$@"  # Pass any additional arguments

