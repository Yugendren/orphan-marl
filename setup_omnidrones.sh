#!/bin/bash
# OmniDrones Setup Script for Drone RL Training
# Requires: Isaac Sim 4.1.0, Linux, NVIDIA GPU

set -e

echo "=========================================="
echo "OmniDrones Setup Script"
echo "=========================================="

# Check ISAACSIM_PATH
if [ -z "$ISAACSIM_PATH" ]; then
    echo "ERROR: ISAACSIM_PATH not set"
    echo "Add to ~/.bashrc:"
    echo '  export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0"'
    exit 1
fi

echo "[1/4] Installing system dependencies..."
sudo apt install -y cmake build-essential tmuxp

echo "[2/4] Cloning Isaac Lab..."
if [ ! -d "IsaacLab" ]; then
    git clone git@github.com:isaac-sim/IsaacLab.git
    cd IsaacLab
    pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash
    ./isaaclab.sh --install
    cd ..
else
    echo "  Isaac Lab already exists, skipping..."
fi

echo "[3/4] Cloning OmniDrones..."
if [ ! -d "OmniDrones" ]; then
    git clone https://github.com/btx0424/OmniDrones.git
    cd OmniDrones
    pip install -e .
    cd ..
else
    echo "  OmniDrones already exists, skipping..."
fi

echo "[4/4] Setting up conda environment scripts..."
if [ -n "$CONDA_PREFIX" ]; then
    cp -r OmniDrones/conda_setup/etc $CONDA_PREFIX
    echo "  Conda activation scripts installed"
else
    echo "  WARNING: Not in conda environment, skipping conda setup"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Re-activate conda environment: conda activate <env>"
echo "  2. Verify installation:"
echo "     cd OmniDrones/scripts"
echo "     python train.py task=Hover algo=ppo headless=true total_frames=1000 wandb.mode=disabled"
echo ""
