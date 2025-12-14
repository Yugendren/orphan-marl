# Quick Start - Get Isaac Gym

## The Issue
Your training code needs **Isaac Gym**, which must be downloaded from NVIDIA (it's not available via pip).

## Quick Solution (3 Steps)

### Step 1: Download Isaac Gym
1. Go to: **https://developer.nvidia.com/isaac-gym**
2. Sign in (create account if needed)
3. Download **"Isaac Gym Preview"** for Linux
   - File name: `isaacgym_preview_linux.tar.gz`

### Step 2: Upload to EC2
From your Mac terminal:
```bash
scp isaacgym_preview_linux.tar.gz ubuntu@YOUR-EC2-IP:/home/ubuntu/
```

### Step 3: Install (Automatic)
On your EC2 instance, run:
```bash
cd /home/ubuntu/Orphan_MVP1
./get_isaac_gym.sh /home/ubuntu/isaacgym_preview_linux.tar.gz
```

That's it! After installation, you can run training.

## Alternative: If You Already Have the File

If you already have `isaacgym_preview_linux.tar.gz` somewhere on EC2, just tell me the path and I'll install it!

## After Installation

Once Isaac Gym is installed, run:
```bash
cd /home/ubuntu/Orphan_MVP1
/opt/IsaacSim/python.sh scripts/train_ppo.py \
    --task-config configs/task/DroneTankTask.yaml \
    --train-config configs/train/DroneTankPPO.yaml \
    --output-dir outputs
```

## Need Help?

- **Don't have NVIDIA account?** Create one at https://developer.nvidia.com/
- **File already on EC2?** Tell me the path: `/path/to/isaacgym_preview_linux.tar.gz`
- **Having issues?** Check the error messages and let me know!

