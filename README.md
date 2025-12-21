# OmniDrones Drone RL Training

GPU-parallelized reinforcement learning for drone navigation to a tank target using [OmniDrones](https://github.com/btx0424/OmniDrones) on NVIDIA Isaac Sim.

## Features
- **GPU-parallelized**: 1000s of environments simultaneously
- **Tank target**: Drone navigates to tank loaded from USD
- **Circular spawn**: Drone spawns on circle around tank, always facing it
- **PPO training**: Built-in PPO/MAPPO with TorchRL
- **Hydra config**: Flexible configuration management

## Requirements
- **Ubuntu 20.04/22.04** (Windows not supported)
- **NVIDIA GPU** (RTX 2080+)
- **Isaac Sim 4.1.0**

## EC2 Installation

```bash
# SSH to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone the repo
cd /home/ubuntu
git clone https://github.com/YOUR_REPO/Orphan_MVP1.git
cd Orphan_MVP1
git checkout omnidrone_test

# Set Isaac Sim path (adjust version as needed)
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0"
export ISAACSIM_PYTHON="${ISAACSIM_PATH}/python.sh"

# Run setup
chmod +x setup_omnidrones.sh
./setup_omnidrones.sh
```

## Usage

### Quick Test with Built-in Hover Task
```bash
cd OmniDrones/scripts
$ISAACSIM_PYTHON train.py task=Hover algo=ppo headless=true total_frames=1000 wandb.mode=disabled
```

### Train Custom DroneToTarget Task
```bash
$ISAACSIM_PYTHON train_omnidrones.py headless=true task=DroneToTarget
```

---

## Spawn Logic Explained

The drone spawns on a **circle around the tank**, always **facing** the tank (camera pointed at target).

### Top-Down View
```
                    N (y+)
                      │
        D2 ─────────  │  ───────── D1
       (θ=135°)       │          (θ=45°)
              ╲       │       ╱
               ╲      │      ╱
                ╲     │     ╱
                 ╲    │    ╱    radius
                  ╲   │   ╱     (8-15m)
         ─────────[TANK]─────────── E (x+)
                  ╱   │   ╲
                 ╱    │    ╲
                ╱     │     ╲
               ╱      │      ╲
        D3 ──╱        │        ╲── D4
       (θ=225°)       │          (θ=315°)
                      │
                    S (y-)
```

### Spawn Algorithm

```python
# Random angle on circle (0 to 2π)
theta = random() * 2π

# Random radius (8-15 meters from tank)
radius = uniform(8.0, 15.0)

# Random height (2-5 meters)
height = uniform(2.0, 5.0)

# Compute position
x = tank_x + radius * cos(theta)
y = tank_y + radius * sin(theta)
z = height

# Face the tank (camera pointing at target)
yaw = atan2(tank_y - y, tank_x - x)
```

### Code Implementation

See [omni_drones/tasks/drone_to_target.py](omni_drones/tasks/drone_to_target.py):

```python
def compute_circular_spawn_position(n_envs, target_position, radius_min, radius_max, ...):
    # Random angle for each environment
    theta = torch.rand(n_envs) * 2 * math.pi
    
    # Random radius
    radius = torch.rand(n_envs) * (radius_max - radius_min) + radius_min
    
    # Position on circle
    positions[:, 0] = target_x + radius * torch.cos(theta)
    positions[:, 1] = target_y + radius * torch.sin(theta)
    positions[:, 2] = random_height
    
    # Yaw to face target
    dx = target_x - positions[:, 0]
    dy = target_y - positions[:, 1]
    yaw = torch.atan2(dy, dx)
    
    return positions, yaw
```

---

## Drone Models

OmniDrones provides **built-in drone models** (no external USD needed):

| Model | Description |
|-------|-------------|
| `Iris` | PX4 standard quadrotor (default) |
| `Hummingbird` | Agile racing drone |
| `Crazyflie` | Small nano drone |

The drone USD is bundled with OmniDrones in:
```
OmniDrones/omni_drones/robots/assets/
```

---

## Configuration

### Task Config: `cfg/task/DroneToTarget.yaml`

```yaml
# Drone spawns on circle around tank
spawn:
  radius_min: 8.0      # Min distance from tank
  radius_max: 15.0     # Max distance from tank
  height_min: 2.0      # Min spawn height
  height_max: 5.0      # Max spawn height
  face_target: true    # Camera always points at tank

# Tank (target)
target:
  usd_path: "scenes/training_scene_simple.usd"
  position: [0.0, 0.0, 0.0]
```

---

## Project Structure

```
├── cfg/
│   ├── train.yaml                 # Main config
│   ├── algo/ppo.yaml              # PPO hyperparameters
│   └── task/DroneToTarget.yaml    # Task config
├── omni_drones/
│   └── tasks/
│       └── drone_to_target.py     # Custom task with spawn logic
├── scenes/
│   └── training_scene_simple.usd  # Tank scene
├── train_omnidrones.py            # Training script
└── setup_omnidrones.sh            # EC2 installation
```
