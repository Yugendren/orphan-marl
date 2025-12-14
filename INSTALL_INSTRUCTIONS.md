# Installation Instructions

## Current Status

Your code uses **Isaac Gym** API (`isaacgym`), but Isaac Gym needs to be downloaded manually from NVIDIA.

## Option 1: Install Isaac Gym (Recommended - Matches Current Code)

Isaac Gym must be downloaded from NVIDIA Developer website:

1. **Download Isaac Gym:**
   - Go to: https://developer.nvidia.com/isaac-gym
   - Sign in with NVIDIA account
   - Download "Isaac Gym Preview" (isaacgym_*.tar.gz file)

2. **Upload to EC2:**
   ```bash
   # If you have the file locally, upload it:
   scp isaacgym_*.tar.gz ubuntu@your-ec2-ip:/home/ubuntu/
   ```

3. **Install on EC2:**
   ```bash
   cd /home/ubuntu
   tar -xzf isaacgym_*.tar.gz
   cd isaacgym/python
   /opt/IsaacSim/python.sh -m pip install -e .
   ```

4. **Verify installation:**
   ```bash
   /opt/IsaacSim/python.sh -c "import isaacgym; print('Isaac Gym installed!')"
   ```

## Option 2: Use Isaac Lab (Requires Code Update)

Isaac Lab is already cloned at `/home/ubuntu/IsaacLab`. However, your code would need to be updated to use Isaac Lab APIs instead of Isaac Gym APIs.

## Quick Test

After installing Isaac Gym, run:

```bash
cd /home/ubuntu/Orphan_MVP1
/opt/IsaacSim/python.sh scripts/train_ppo.py \
    --task-config configs/task/DroneTankTask.yaml \
    --train-config configs/train/DroneTankPPO.yaml \
    --output-dir outputs
```

## Need Help?

If you have the Isaac Gym file already, let me know the path and I can install it for you!

