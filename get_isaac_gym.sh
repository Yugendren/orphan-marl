#!/bin/bash
# Helper script to download and install Isaac Gym

echo "=========================================="
echo "Isaac Gym Download & Installation Helper"
echo "=========================================="
echo ""
echo "Isaac Gym Preview must be downloaded from NVIDIA Developer Portal."
echo ""
echo "STEP 1: Download Isaac Gym"
echo "---------------------------"
echo "1. Open this URL in your browser:"
echo "   https://developer.nvidia.com/isaac-gym"
echo ""
echo "2. Sign in with your NVIDIA Developer account"
echo "   (Create one at https://developer.nvidia.com/ if needed)"
echo ""
echo "3. Download 'Isaac Gym Preview' for Linux"
echo "   (File will be named: isaacgym_preview_linux.tar.gz)"
echo ""
echo "STEP 2: Upload to EC2"
echo "---------------------"
echo "From your local machine, run:"
echo "  scp isaacgym_preview_linux.tar.gz ubuntu@YOUR-EC2-IP:/home/ubuntu/"
echo ""
echo "STEP 3: Install (I'll do this automatically)"
echo "--------------------------------------------"
echo "Once uploaded, run this script again with the file path:"
echo "  ./get_isaac_gym.sh /home/ubuntu/isaacgym_preview_linux.tar.gz"
echo ""

# If file path provided, install it
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Installing Isaac Gym from: $1"
    echo ""
    
    # Extract
    cd /home/ubuntu
    tar -xzf "$1"
    
    # Find isaacgym directory
    ISAACGYM_DIR=$(find /home/ubuntu -type d -name "isaacgym" | head -1)
    
    if [ -z "$ISAACGYM_DIR" ]; then
        echo "ERROR: Could not find isaacgym directory after extraction"
        exit 1
    fi
    
    echo "Found Isaac Gym at: $ISAACGYM_DIR"
    cd "$ISAACGYM_DIR/python"
    
    # Install
    echo "Installing Isaac Gym..."
    /opt/IsaacSim/python.sh -m pip install -e .
    
    # Verify
    echo ""
    echo "Verifying installation..."
    /opt/IsaacSim/python.sh -c "import isaacgym; print('✓ Isaac Gym installed successfully!')" && echo "Installation complete!" || echo "Installation failed - check errors above"
fi

