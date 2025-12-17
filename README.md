# Drone RL Training Script (`train_drone_ppo.py`)

## Description
This script implements Reinforcement Learning (RL) training for a quadrotor drone using Proximal Policy Optimization (PPO). It runs within the **NVIDIA Isaac Sim** environment using the **Pegasus Simulator** extension.

The drone's objective is to navigate to a target position (visualized as a red cube) in 3D space.

## Dependencies
- Python 3.10 (Isaac Sim python)
- NVIDIA Isaac Sim
- Pegasus Simulator Extension
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- Gymnasium

## Usage

### Training
Run the training loop. By default, the script runs in **training mode** (`--mode train`), so you only need to specify other parameters like timesteps.

**Standard Training (State-based observations):**
```bash
# Basic training (default: 100k steps)
$ISAACSIM_PYTHON train_drone_ppo.py

# Train for specific number of timesteps
$ISAACSIM_PYTHON train_drone_ppo.py --timesteps 100000

# Train with GUI enabled (to see the drone)
$ISAACSIM_PYTHON train_drone_ppo.py --gui
```

**Vision-based Training:**
To train using camera input (160x120 RGB images):
```bash
$ISAACSIM_PYTHON train_drone_ppo.py --vision --timesteps 100000
```

> [!IMPORTANT]
> **Headless Vision Training**:
> Training with vision (`--vision`) requires a comprehensive rendering context (OpenGL/Vulkan). In headless environments (like Docker or remote servers), the script will crash if no display is found.
>
> To fix this, you must either:
> 1. Run with `--gui` if you have a display.
> 2. Use **Xvfb** (Virtual Framebuffer) to provide a fake display:
>    ```bash
>    # Install Xvfb if needed
>    # sudo apt-get install xvfb
>
>    xvfb-run -a $ISAACSIM_PYTHON train_drone_ppo.py --vision
>    ```

### Testing
Evaluate a trained model.

```bash
$ISAACSIM_PYTHON train_drone_ppo.py --mode test --model-path ./drone_ppo_models/drone_ppo_final.zip
```

## Command Line Arguments
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `train` | Mode to run: `train` or `test`. Optional for training. |
| `--vision`| flag | `False` | Application of Vision-based observations (160x120 RGB). If omitted, uses State-based. |
| `--timesteps`| int | `100_000` | Total number of training timesteps. |
| `--model-path`| str | `None` | Path to a `.zip` model file. Required when mode is `test`. |
| `--gui` | flag | `False` | Run with the simulator GUI window open (disables headless mode). |

## Environment Details

### Observation Space
1. **State-based (Default)**: A 9-dimensional vector containing:
   - Drone Position `[x, y, z]`
   - Drone Linear Velocity `[vx, vy, vz]`
   - Relative Target Position `[tx-x, ty-y, tz-z]`

2. **Vision-based**:
   - Shape: `(120, 160, 3)`
   - Type: RGB Image representing the drone's front-facing camera view.

### Action Space
Continuous control vector of size 4:
- `[vx, vy, vz, yaw_rate]`
- Range: `vx, vy` in [-2, 2], `vz, yaw_rate` in [-1, 1].
- These are velocity commands sent to a lower-level velocity controller.

### Reward Function
The agent receives rewards based on:
- **Distance**: Negative Euclidean distance to the target (closer is better).
- **Success**: `+100` bonus for reaching within 0.5m of the target.
- **Crash**: `-50` penalty for hitting the ground (z < 0.1).
- **Out of Bounds**: `-50` penalty for moving too far (> 15m) from target.

## Output
- **Logs**: TensorBoard logs are saved to `./drone_ppo_models/logs`.
- **Checkpoints**: Models are saved every 10,000 steps to `./drone_ppo_models/`.
- **Final Model**: Saved as `drone_ppo_final.zip` upon completion.
