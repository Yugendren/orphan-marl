#!/bin/bash
# Setup Isaac Lab environment for training

export ISAACLAB_PATH="/home/ubuntu/IsaacLab"
export ISAACSIM_PYTHON="/opt/IsaacSim/python.sh"

# Add Isaac Lab to Python path
export PYTHONPATH="${ISAACLAB_PATH}/source:${PYTHONPATH}"

echo "Isaac Lab environment setup complete!"
echo "Isaac Lab path: $ISAACLAB_PATH"
echo "Isaac Sim Python: $ISAACSIM_PYTHON"
echo ""
echo "To use, run:"
echo "  source setup_isaac_lab.sh"
echo "  /opt/IsaacSim/python.sh scripts/train_ppo.py ..."

