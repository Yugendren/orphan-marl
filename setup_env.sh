#!/bin/bash
# Setup script for Isaac Sim RL training environment

echo "Setting up Isaac Sim RL environment..."

# Check if Isaac Sim Python is available
if [ -z "$ISAACSIM_PYTHON" ]; then
    echo "ISAACSIM_PYTHON not set. Trying to find Isaac Sim..."
    if [ -f "/opt/IsaacSim/python.sh" ]; then
        export ISAACSIM_PYTHON="/opt/IsaacSim/python.sh"
        echo "Found Isaac Sim at: $ISAACSIM_PYTHON"
    else
        echo "ERROR: Isaac Sim not found. Please install Isaac Sim first."
        exit 1
    fi
fi

echo "Using Isaac Sim Python: $ISAACSIM_PYTHON"

# Install/upgrade pip
$ISAACSIM_PYTHON -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
$ISAACSIM_PYTHON -m pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run training, use:"
echo "  $ISAACSIM_PYTHON scripts/train_ppo.py --task-config configs/task/DroneTankTask.yaml --train-config configs/train/DroneTankPPO.yaml --output-dir outputs"
echo ""
echo "Or activate the environment:"
echo "  export ISAACSIM_PYTHON=$ISAACSIM_PYTHON"
echo "  alias python='$ISAACSIM_PYTHON'"

