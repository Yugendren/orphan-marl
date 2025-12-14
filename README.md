# Orphan MVP1 - Drone Tank RL Task

A reinforcement learning project for Isaac Sim/Lab where a drone learns to identify and strike a tank using computer vision. The drone uses camera observations to navigate towards the tank, receiving positive rewards for successful hits and negative rewards for crashing into the ground.

## Overview

This project implements a complete RL training pipeline for a drone agent that must:
- Use computer vision (camera images) to detect and identify a tank
- Navigate towards the tank using visual and state information
- Strike/touch the tank to receive positive rewards
- Avoid crashing into the ground (negative reward)

## Project Structure

```
Orphan_MVP1/
├── configs/                    # Configuration files
│   ├── task/                  # Task-specific configurations
│   │   └── DroneTankTask.yaml # Environment and task settings
│   └── train/                 # Training configurations
│       └── DroneTankPPO.yaml  # PPO training hyperparameters
│
├── environments/              # Environment implementations
│   └── drone_tank_task.py    # Main task environment (Isaac Sim/Lab)
│
├── agents/                    # RL agent implementations
│   └── drone_agent.py        # Actor-Critic network with vision
│
├── rewards/                   # Reward function definitions
│   └── reward_functions.py   # Tank hit, crash penalty, distance rewards
│
├── vision/                    # Computer vision processing
│   └── vision_processor.py   # CNN encoder for camera images
│
├── utils/                     # Utility functions
│   └── env_utils.py          # Environment helpers and utilities
│
├── scripts/                   # Training and evaluation scripts
│   ├── train.py              # Main training script
│   └── evaluate.py           # Evaluation script for trained models
│
├── assets/                    # USD assets (to be added)
│   ├── drone.usd             # Drone 3D model
│   └── tank.usd              # Tank 3D model
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

### Environment
- **Drone Agent**: Controllable drone with physics simulation
- **Tank Target**: USD asset representing the target tank
- **Camera Sensor**: RGB camera attached to drone for vision-based navigation
- **State Observations**: Position, velocity, orientation, angular velocity
- **Vision Observations**: RGB camera images for CV-based detection

### Reward System
- **Tank Hit Reward** (+10.0): Positive reward when drone successfully touches the tank
- **Ground Crash Penalty** (-5.0): Negative reward when drone crashes into the ground
- **Distance Reward**: Encourages getting closer to the tank
- **Survival Reward**: Small positive reward for staying alive

### RL Agent
- **Actor-Critic Architecture**: Separate policy and value networks
- **Vision Integration**: CNN encoder processes camera images
- **State + Vision Fusion**: Combines visual and proprioceptive information
- **Continuous Action Space**: 4D action space (thrust in x, y, z + yaw)

### Vision Processing
- **CNN Encoder**: Multi-layer convolutional network for feature extraction
- **Image Preprocessing**: Normalization and resizing
- **Feature Extraction**: 256-dimensional feature vectors from RGB images

## Setup

### Prerequisites

1. **Isaac Sim/Lab**: Install Isaac Sim or Isaac Lab
   - Download from NVIDIA Omniverse
   - Follow official installation instructions

2. **Python Environment**: Python 3.8+ recommended

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Orphan_MVP1
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install Isaac Sim/Lab dependencies:
   - Follow Isaac Sim/Lab installation guide
   - Ensure `isaacgym` and `isaacgymenvs` are properly installed

4. Add USD Assets:
   - Place your drone USD file at `assets/drone.usd`
   - Place your tank USD file at `assets/tank.usd`
   - If assets are not available, the code will create placeholder boxes

## Configuration

### Task Configuration (`configs/task/DroneTankTask.yaml`)

Key settings:
- **Drone spawn position**: Initial drone location
- **Tank spawn position**: Target tank location
- **Camera settings**: Resolution, FOV, position
- **Observation space**: State and vision observation settings
- **Action space**: Continuous action dimensions and scaling
- **Episode settings**: Max length, reset conditions

### Training Configuration (`configs/train/DroneTankPPO.yaml`)

Key settings:
- **Number of environments**: Parallel environments for training (default: 4096)
- **PPO hyperparameters**: Learning rate, gamma, clip param, etc.
- **Network architecture**: MLP and CNN layer configurations
- **Training settings**: Max iterations, checkpoint intervals

### Reward Configuration

Reward parameters can be adjusted in the task configuration:
```yaml
reward:
  tank_hit: 10.0              # Reward for hitting tank
  ground_crash: -5.0          # Penalty for crashing
  distance_scale: 0.1         # Distance reward scaling
  survival: 0.01              # Survival reward
  tank_contact_threshold: 0.5  # Distance threshold for tank hit
  ground_contact_threshold: 0.3  # Distance threshold for crash
```

## Usage

### Training

Train a new agent from scratch:
```bash
python scripts/train.py \
    --task-config configs/task/DroneTankTask.yaml \
    --train-config configs/train/DroneTankPPO.yaml \
    --output-dir outputs
```

Resume training from a checkpoint:
```bash
python scripts/train.py \
    --task-config configs/task/DroneTankTask.yaml \
    --train-config configs/train/DroneTankPPO.yaml \
    --checkpoint outputs/checkpoint_5000.pt \
    --output-dir outputs
```

Run in headless mode (no GUI):
```bash
python scripts/train.py --headless
```

### Evaluation

Evaluate a trained agent:
```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_10000.pt \
    --num-episodes 10 \
    --render
```

Evaluate without rendering (faster):
```bash
python scripts/evaluate.py \
    --checkpoint outputs/checkpoint_10000.pt \
    --num-episodes 100
```

## Architecture

### Environment Flow

1. **Initialization**: Spawn drone and tank in simulation
2. **Observation**: Collect state (position, velocity) and vision (camera images)
3. **Action**: Agent selects action (thrust forces)
4. **Physics Step**: Simulate drone movement
5. **Reward Calculation**: 
   - Check distance to tank (positive if close)
   - Check distance to ground (negative if too close)
   - Compute combined reward
6. **Reset**: Reset environment if crash or success

### Agent Architecture

```
Input: [State (13D) + Vision Features (256D)]
    ↓
Shared MLP: [512, 512, 256]
    ↓
    ├─→ Actor Head → Action Mean (4D)
    │                Action Std (4D)
    │
    └─→ Critic Head → Value Estimate (1D)
```

### Vision Pipeline

```
Camera Image (640x480x3 RGB)
    ↓
CNN Encoder:
    Conv2d(32) → Conv2d(64) → Conv2d(128) → Conv2d(256)
    ↓
Flatten + FC Layers
    ↓
Vision Features (256D)
```

## Reward Details

### Tank Hit Reward
- **Trigger**: Drone within `tank_contact_threshold` distance of tank
- **Value**: +10.0 (configurable)
- **Purpose**: Encourage successful strikes

### Ground Crash Penalty
- **Trigger**: Drone within `ground_contact_threshold` of ground
- **Value**: -5.0 (configurable)
- **Purpose**: Discourage crashes

### Distance Reward
- **Formula**: `-distance_to_tank * distance_scale`
- **Purpose**: Encourage getting closer to target

### Survival Reward
- **Value**: +0.01 per timestep
- **Purpose**: Encourage exploration and prevent premature termination

## Customization

### Adding Custom Reward Functions

1. Create a new reward class in `rewards/reward_functions.py`:
```python
class CustomReward(RewardFunction):
    def compute(self, **kwargs):
        # Your reward logic
        return reward_tensor
```

2. Add it to the combined reward in `scripts/train.py`

### Modifying Vision Encoder

Edit `vision/vision_processor.py` to:
- Change CNN architecture
- Add different preprocessing
- Implement object detection
- Add depth estimation

### Changing Action Space

Modify `configs/task/DroneTankTask.yaml`:
```yaml
actions:
  num_actions: 6  # Change from 4 to 6 for more control
  scale: [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
```

## Troubleshooting

### Common Issues

1. **Isaac Sim not found**: Ensure Isaac Sim/Lab is properly installed and paths are set
2. **USD assets not loading**: Check file paths in config, or let code create placeholders
3. **CUDA out of memory**: Reduce `num_envs` in training config
4. **Import errors**: Ensure all dependencies are installed and paths are correct

### Performance Tips

- Use GPU pipeline for faster simulation
- Adjust `num_envs` based on available GPU memory
- Use headless mode for faster training
- Reduce image resolution if vision processing is slow

## Future Enhancements

- [ ] Integration with RL-Games framework
- [ ] Advanced object detection (YOLO, etc.)
- [ ] Multi-tank scenarios
- [ ] Obstacle avoidance
- [ ] Real-world deployment considerations
- [ ] TensorBoard logging
- [ ] Hyperparameter tuning scripts

## License

[Add your license here]

## Contributors

[Add contributors here]

## Acknowledgments

- NVIDIA Isaac Sim/Lab for the simulation framework
- RL-Games for RL algorithm implementations (if used)
