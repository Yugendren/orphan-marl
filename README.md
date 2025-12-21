# OmniDrones Drone RL Training

GPU-parallelized reinforcement learning for drone navigation using [OmniDrones](https://github.com/btx0424/OmniDrones) on NVIDIA Isaac Sim.

## Features
- **GPU-parallelized**: 1000s of environments running simultaneously
- **PPO training**: Built-in PPO/MAPPO with TorchRL
- **Custom task**: DroneToTarget navigation with reward shaping
- **Hydra config**: Flexible configuration management
- **WandB logging**: Experiment tracking and video recording

## Requirements
- Ubuntu 20.04/22.04 (Windows not supported)
- NVIDIA GPU (RTX 2080+ recommended)
- Isaac Sim 4.1.0
- Isaac Lab

## Installation

```bash
# Set Isaac Sim path
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0"

# Run setup script
chmod +x setup_omnidrones.sh
./setup_omnidrones.sh
```

## Usage

### Quick Test with Built-in Hover Task
```bash
cd OmniDrones/scripts
python train.py task=Hover algo=ppo headless=true total_frames=1000 wandb.mode=disabled
```

### Train Custom DroneToTarget Task
```bash
python train_omnidrones.py headless=true task=DroneToTarget
```

### With Evaluation and Checkpointing
```bash
python train_omnidrones.py headless=true eval_interval=100 save_interval=500
```

### Enable WandB Logging
```bash
python train_omnidrones.py wandb.mode=online wandb.entity=YOUR_USERNAME
```

## Configuration

Configurations are managed with Hydra:

| File | Description |
|------|-------------|
| `cfg/train.yaml` | Main training config |
| `cfg/algo/ppo.yaml` | PPO hyperparameters |
| `cfg/task/DroneToTarget.yaml` | Task-specific settings |

### Override Config from Command Line
```bash
python train_omnidrones.py task.env.num_envs=2048 algo.ppo_epochs=8
```

## Task: DroneToTarget

Navigate a drone to randomly placed targets.

**Observation Space (13D):**
- Drone position [3]
- Linear velocity [3]
- Angular velocity [3]
- Relative target [3]
- Distance [1]

**Action Space (3D):**
- Velocity commands [vx, vy, vz]

**Reward:**
- Distance penalty: `-1.0 * distance`
- Progress reward: `+20.0 * (prev_distance - distance)`
- Success bonus: `+100.0`
- Crash penalty: `-50.0`

## Project Structure

```
├── cfg/
│   ├── train.yaml           # Main config
│   ├── algo/
│   │   └── ppo.yaml          # PPO config
│   └── task/
│       └── DroneToTarget.yaml # Task config
├── omni_drones/
│   └── tasks/
│       └── drone_to_target.py # Custom task
├── train_omnidrones.py       # Training script
├── setup_omnidrones.sh       # Installation script
└── README.md
```

## Troubleshooting

**ISAACSIM_PATH not set:**
```bash
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0"
```

**Task not found:**
Ensure custom task is registered by importing:
```python
from omni_drones.tasks import DroneToTargetTask
```

## Citation

If you use this code, please cite OmniDrones:
```bibtex
@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control},
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv}
}
```
