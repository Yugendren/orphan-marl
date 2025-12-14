#!/bin/bash
# Script to help download and install Isaac Gym

echo "=========================================="
echo "Isaac Gym Installation Guide"
echo "=========================================="
echo ""
echo "Isaac Gym needs to be downloaded manually from NVIDIA."
echo ""
echo "Steps:"
echo "1. Go to: https://developer.nvidia.com/isaac-gym"
echo "2. Sign in with your NVIDIA account"
echo "3. Download 'Isaac Gym Preview' (isaacgym_*.tar.gz)"
echo "4. Upload it to this EC2 instance"
echo ""
echo "Once you have the file, run:"
echo "  tar -xzf isaacgym_*.tar.gz"
echo "  cd isaacgym/python"
echo "  /opt/IsaacSim/python.sh -m pip install -e ."
echo ""
echo "OR if you already have it downloaded, tell me the path and I'll install it."
echo ""

