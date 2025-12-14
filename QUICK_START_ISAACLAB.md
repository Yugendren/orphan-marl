# Quick Start - Isaac Lab Migration Complete! 🎉

## ✅ Migration Complete

Your code has been successfully migrated from deprecated **Isaac Gym** to modern **Isaac Lab** APIs!

## 🚀 How to Run Training

### Simple Method (Recommended)

```bash
cd /home/ubuntu/Orphan_MVP1
./run_training.sh
```

### Manual Method

```bash
cd /home/ubuntu/Orphan_MVP1

# Set up Isaac Lab path
export PYTHONPATH="/home/ubuntu/IsaacLab/source:$PYTHONPATH"

# Run training (with GUI)
/opt/IsaacSim/python.sh scripts/train_ppo_isaaclab.py \
    --num-envs 4096 \
    --output-dir outputs

# Or headless (faster)
/opt/IsaacSim/python.sh scripts/train_ppo_isaaclab.py \
    --num-envs 4096 \
    --output-dir outputs \
    --headless
```

## 📋 What Changed

### ✅ New Files (Use These!)
- `environments/drone_tank_task_isaaclab.py` - Isaac Lab environment
- `scripts/train_ppo_isaaclab.py` - Isaac Lab training script

### ⚠️ Old Files (Deprecated)
- `environments/drone_tank_task.py` - Old Isaac Gym version
- `scripts/train_ppo.py` - Old Isaac Gym version

## 🎯 Key Features

✅ **Proper RL Training**: Full PPO with experience buffers, GAE, policy updates  
✅ **CV Support**: Camera configured (currently state-only, can enable vision)  
✅ **Isaac Lab APIs**: Modern, maintained APIs  
✅ **Works with Isaac Sim**: Properly integrated  

## 📊 Expected Results

- **Training time**: ~1.5-5 hours for 10,000 iterations
- **Rewards**: Should increase from negative to positive over time
- **Checkpoints**: Saved every 500 iterations in `outputs/` directory

## 🔧 Configuration

The environment configuration is now in `environments/drone_tank_task_isaaclab.py` using `@configclass`:

- **Drone**: Uses your USD file at `/home/ubuntu/isaac_assets/SM_Tank_T72b.usd`
- **Tank**: Uses `assets/tank.usd` (or creates placeholder)
- **Camera**: 640x480 RGB camera attached to drone
- **Rewards**: Tank hit (+10.0), ground crash (-5.0), distance, survival

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'isaaclab'"
**Solution**: Make sure Isaac Lab path is set:
```bash
export PYTHONPATH="/home/ubuntu/IsaacLab/source:$PYTHONPATH"
```

### Issue: "No module named 'omni.physics'"
**Solution**: This is normal - Isaac Lab needs to be run through AppLauncher (which the script does)

### Issue: Training is slow
**Solution**: 
- Use `--headless` flag
- Reduce `--num-envs` (e.g., `--num-envs 1024`)

## 📝 Next Steps

1. **Run training**: Use the command above
2. **Monitor progress**: Watch the metrics printed every 10 iterations
3. **Check checkpoints**: Look in `outputs/` directory
4. **Evaluate**: Use evaluation script to test trained models

## 🎓 What You'll See

```
Iteration 0:
  Avg Reward: -2.3456
  Avg Advantage: 0.1234
  Policy Loss: 0.5678
  Value Loss: 0.2345
  Entropy: 1.2345
```

As training progresses, rewards should increase and losses should decrease!

---

**Ready to train!** Just run `./run_training.sh` and watch your drone learn! 🚁🎯

