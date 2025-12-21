#!/bin/bash
# OmniDrones Setup Script for EC2 with Isaac Sim
# Assumes: Isaac Sim and Pegasus Simulator already installed

set -e

echo "=========================================="
echo "OmniDrones Setup Script (EC2)"
echo "=========================================="

# ============================================
# STEP 1: Set environment variables
# ============================================
echo "[1/5] Checking Isaac Sim environment..."

if [ -z "$ISAACSIM_PATH" ]; then
    # Try to find Isaac Sim
    ISAACSIM_PATH=$(find ~/.local/share/ov/pkg/ -maxdepth 1 -name "isaac-sim-*" -type d 2>/dev/null | head -1)
    
    if [ -z "$ISAACSIM_PATH" ]; then
        echo "ERROR: Isaac Sim not found. Set ISAACSIM_PATH manually."
        echo "  export ISAACSIM_PATH=~/.local/share/ov/pkg/isaac-sim-4.1.0"
        exit 1
    fi
    
    echo "  Found Isaac Sim: $ISAACSIM_PATH"
    echo "  Add to ~/.bashrc:"
    echo "    export ISAACSIM_PATH=\"$ISAACSIM_PATH\""
else
    echo "  ISAACSIM_PATH=$ISAACSIM_PATH"
fi

# Use Isaac Sim's Python
export ISAACSIM_PYTHON="${ISAACSIM_PATH}/python.sh"

# ============================================
# STEP 2: Install system dependencies
# ============================================
echo ""
echo "[2/5] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y cmake build-essential tmuxp git

# ============================================
# STEP 3: Clone and install Isaac Lab
# ============================================
echo ""
echo "[3/5] Setting up Isaac Lab..."

if [ ! -d "IsaacLab" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    
    # Install Python dependencies using Isaac Sim's Python
    $ISAACSIM_PYTHON -m pip install usd-core==23.11 lxml==4.9.4 tqdm xxhash
    
    # Install Isaac Lab
    ./isaaclab.sh --install
    cd ..
    echo "  ✓ Isaac Lab installed"
else
    echo "  Isaac Lab already exists, skipping..."
fi

# ============================================
# STEP 4: Clone and install OmniDrones
# ============================================
echo ""
echo "[4/5] Setting up OmniDrones..."

if [ ! -d "OmniDrones" ]; then
    git clone https://github.com/btx0424/OmniDrones.git
    cd OmniDrones
    
    # Install using Isaac Sim's Python
    $ISAACSIM_PYTHON -m pip install -e .
    cd ..
    echo "  ✓ OmniDrones installed"
else
    echo "  OmniDrones already exists, skipping..."
fi

# ============================================
# STEP 5: Setup conda environment (optional)
# ============================================
echo ""
echo "[5/5] Conda setup (optional)..."

if [ -n "$CONDA_PREFIX" ]; then
    if [ -d "OmniDrones/conda_setup/etc" ]; then
        cp -r OmniDrones/conda_setup/etc $CONDA_PREFIX
        echo "  ✓ Conda activation scripts installed"
        echo "  Re-activate conda: conda activate $CONDA_DEFAULT_ENV"
    fi
else
    echo "  Not in conda environment, using Isaac Sim Python directly"
fi

# ============================================
# DONE
# ============================================
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Add these to ~/.bashrc:"
echo "  export ISAACSIM_PATH=\"$ISAACSIM_PATH\""
echo "  export ISAACSIM_PYTHON=\"\${ISAACSIM_PATH}/python.sh\""
echo ""
echo "Verify installation:"
echo "  cd OmniDrones/scripts"
echo "  \$ISAACSIM_PYTHON train.py task=Hover algo=ppo headless=true total_frames=1000 wandb.mode=disabled"
echo ""
echo "Run custom DroneToTarget task:"
echo "  \$ISAACSIM_PYTHON train_omnidrones.py headless=true"
echo ""
