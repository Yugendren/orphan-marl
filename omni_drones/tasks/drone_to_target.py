"""
DroneToTarget Task for OmniDrones

Navigate a drone to a tank target. The drone spawns randomly on a circle 
around the tank, always facing it (camera pointed at target).

Spawn Logic:
    - Random angle θ on circle around tank
    - Random radius within [radius_min, radius_max]
    - Random height within [height_min, height_max]
    - Drone orientation: yaw = atan2(tank_y - drone_y, tank_x - drone_x)

Observation Space:
    - Drone position [3]
    - Drone linear velocity [3]
    - Drone angular velocity [3]
    - Relative target position [3]
    - Distance to target [1]
    Total: 13-dimensional state vector

Action Space:
    - Velocity commands [vx, vy, vz] in drone body frame

Reward:
    - Distance-based penalty
    - Progress reward for getting closer
    - Success bonus for reaching target
    - Crash penalty
    - Facing bonus for keeping target in camera view
"""

import math
import torch
import numpy as np
from typing import Dict, Optional, Tuple

from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec

# OmniDrones imports - these will be available after OmniDrones is installed
try:
    from omni_drones.envs.isaac_env import IsaacEnv
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.envs import register_env
    OMNIDRONES_AVAILABLE = True
except ImportError:
    OMNIDRONES_AVAILABLE = False
    print("[WARNING] OmniDrones not installed. Install with: pip install -e OmniDrones/")
    
    # Placeholder for development
    class IsaacEnv:
        REGISTRY = {}
    def register_env(name):
        def decorator(cls):
            return cls
        return decorator


def compute_circular_spawn_position(
    n_envs: int,
    target_position: torch.Tensor,
    radius_min: float,
    radius_max: float,
    height_min: float,
    height_max: float,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spawn positions on a circle around the target.
    
    Args:
        n_envs: Number of environments
        target_position: Target (tank) position [3]
        radius_min: Minimum spawn radius
        radius_max: Maximum spawn radius
        height_min: Minimum spawn height
        height_max: Maximum spawn height
        device: Torch device
        
    Returns:
        positions: Spawn positions [n_envs, 3]
        yaw_angles: Yaw angles to face target [n_envs]
    
    Diagram (top view):
    
                    N (y+)
                      |
                      |
        W -------- [TANK] -------- E (x+)
                      |
                      |
                    S
        
        Drone spawns at random angle θ:
        
                   D2 (θ=90°)
                  /
                 /  radius
                /
        [TANK] -------- D1 (θ=0°)
                \\
                 \\
                  D3 (θ=270°)
        
        Each drone faces the tank (yaw = θ + 180°)
    """
    # Random angle for each environment (0 to 2π)
    theta = torch.rand(n_envs, device=device) * 2 * math.pi
    
    # Random radius within range
    radius = torch.rand(n_envs, device=device) * (radius_max - radius_min) + radius_min
    
    # Random height within range
    height = torch.rand(n_envs, device=device) * (height_max - height_min) + height_min
    
    # Compute XY positions on circle
    # x = target_x + radius * cos(theta)
    # y = target_y + radius * sin(theta)
    positions = torch.zeros(n_envs, 3, device=device)
    positions[:, 0] = target_position[0] + radius * torch.cos(theta)
    positions[:, 1] = target_position[1] + radius * torch.sin(theta)
    positions[:, 2] = height
    
    # Compute yaw to face target
    # yaw = atan2(target_y - drone_y, target_x - drone_x)
    # This makes the drone's forward direction (camera) point at the tank
    dx = target_position[0] - positions[:, 0]
    dy = target_position[1] - positions[:, 1]
    yaw_angles = torch.atan2(dy, dx)
    
    return positions, yaw_angles


def yaw_to_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert yaw angles to quaternions (w, x, y, z format).
    
    Args:
        yaw: Yaw angles in radians [n_envs]
        
    Returns:
        quaternions: [n_envs, 4] in (w, x, y, z) format
    """
    n = yaw.shape[0]
    quat = torch.zeros(n, 4, device=yaw.device)
    
    # Quaternion from yaw-only rotation (around Z axis)
    half_yaw = yaw * 0.5
    quat[:, 0] = torch.cos(half_yaw)  # w
    quat[:, 1] = 0.0                   # x
    quat[:, 2] = 0.0                   # y
    quat[:, 3] = torch.sin(half_yaw)  # z
    
    return quat


@register_env("DroneToTarget")
class DroneToTargetTask(IsaacEnv):
    """
    Navigate drone to tank target.
    
    Features:
        - Tank loaded from USD file
        - Drone spawns on circle around tank
        - Drone always faces tank (camera pointing at target)
        - GPU-parallelized (1000s of environments)
    """
    
    def __init__(self, cfg, headless: bool):
        """
        Initialize the DroneToTarget task.
        
        Args:
            cfg: Hydra configuration
            headless: Whether to run without GUI
        """
        # Spawn configuration
        spawn_cfg = cfg.task.get("spawn", {})
        self.spawn_radius_min = spawn_cfg.get("radius_min", 8.0)
        self.spawn_radius_max = spawn_cfg.get("radius_max", 15.0)
        self.spawn_height_min = spawn_cfg.get("height_min", 2.0)
        self.spawn_height_max = spawn_cfg.get("height_max", 5.0)
        self.face_target = spawn_cfg.get("face_target", True)
        
        # Target (tank) configuration
        target_cfg = cfg.task.get("target", {})
        self.target_usd_path = target_cfg.get("usd_path", "scenes/training_scene_simple.usd")
        self.target_prim_path = target_cfg.get("prim_path", "/World/SM_Tank_T72b")
        self.target_position_cfg = target_cfg.get("position", [0.0, 0.0, 0.0])
        
        # Reward configuration
        self.reward_cfg = cfg.task.get("reward", {})
        
        # Success threshold
        self.target_threshold = target_cfg.get("distance_threshold", 0.5)
        self.max_distance = spawn_cfg.get("radius_max", 15.0) * 2
        
        super().__init__(cfg, headless)
        
        # Target position tensor
        self.target_position = torch.tensor(
            self.target_position_cfg, device=self.device
        )
        
        # Tracking variables
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)
        self.episode_length = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros(self.num_envs, device=self.device)
        
    def _design_scene(self):
        """
        Design the simulation scene.
        
        Loads:
            - Ground plane
            - Tank from USD file (target)
            - Drone (Iris quadrotor)
        """
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.utils.stage import add_reference_to_stage
        
        # Add ground plane
        prim_utils.create_prim("/World/ground", "Plane")
        
        # Load tank from USD file
        if self.target_usd_path:
            print(f"[DroneToTarget] Loading tank from: {self.target_usd_path}")
            add_reference_to_stage(
                usd_path=self.target_usd_path,
                prim_path="/World/Tank"
            )
        
        # Create drone - OmniDrones provides various drone models
        # The Iris model is a standard quadrotor used in PX4 SITL
        self.drone = MultirotorBase(
            name="drone",
            drone_model=self.cfg.task.drone.get("model", "Iris"),
        )
        
        print(f"[DroneToTarget] Scene ready:")
        print(f"  - Tank at: {self.target_position_cfg}")
        print(f"  - Spawn radius: {self.spawn_radius_min}-{self.spawn_radius_max}m")
        print(f"  - Spawn height: {self.spawn_height_min}-{self.spawn_height_max}m")
    
    def _set_specs(self):
        """Define observation and action specifications."""
        obs_dim = 13  # pos(3) + vel(3) + ang_vel(3) + rel_target(3) + dist(1)
        
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    shape=(self.num_envs, 1, obs_dim),
                    device=self.device
                ),
            }),
            "stats": CompositeSpec({
                "return": UnboundedContinuousTensorSpec((self.num_envs, 1)),
                "episode_len": UnboundedContinuousTensorSpec((self.num_envs, 1)),
                "success": UnboundedContinuousTensorSpec((self.num_envs, 1)),
            }),
        })
        
        # Action: velocity commands [vx, vy, vz]
        action_cfg = self.cfg.task.get("action", {})
        max_vel = action_cfg.get("max_velocity", [2.0, 2.0, 1.0])
        
        self.action_spec = BoundedTensorSpec(
            low=torch.tensor([-max_vel[0], -max_vel[1], -max_vel[2]]),
            high=torch.tensor([max_vel[0], max_vel[1], max_vel[2]]),
            shape=(self.num_envs, 1, 3),
            device=self.device
        )
        
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(self.num_envs, 1), device=self.device
        )
        
        self.done_spec = BoundedTensorSpec(
            low=0, high=1,
            shape=(self.num_envs, 1),
            dtype=torch.bool,
            device=self.device
        )
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Reset specified environments with circular spawn.
        
        The drone spawns on a random position on a circle around the tank,
        with its camera facing the tank.
        """
        n = len(env_ids)
        
        # Compute spawn positions on circle around tank
        positions, yaw_angles = compute_circular_spawn_position(
            n_envs=n,
            target_position=self.target_position,
            radius_min=self.spawn_radius_min,
            radius_max=self.spawn_radius_max,
            height_min=self.spawn_height_min,
            height_max=self.spawn_height_max,
            device=self.device
        )
        
        # Convert yaw to quaternion for orientation
        if self.face_target:
            orientations = yaw_to_quaternion(yaw_angles)
        else:
            # Random orientation
            orientations = torch.zeros(n, 4, device=self.device)
            orientations[:, 0] = 1.0  # Identity quaternion
        
        # Set drone poses
        self.drone.set_world_poses(positions, orientations, env_ids=env_ids)
        
        # Reset velocities
        zeros = torch.zeros(n, 3, device=self.device)
        self.drone.set_velocities(zeros, zeros, env_ids=env_ids)
        
        # Reset tracking
        self.prev_distance[env_ids] = self._compute_distance(env_ids)
        self.episode_length[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.success[env_ids] = 0
    
    def _pre_sim_step(self, tensordict: TensorDict):
        """Apply velocity commands before physics step."""
        actions = tensordict["agents"]["action"].squeeze(1)
        self.drone.apply_velocity_commands(actions)
    
    def _compute_distance(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute distance from drone to target."""
        if env_ids is None:
            drone_pos = self.drone.get_world_poses()[0]
        else:
            drone_pos = self.drone.get_world_poses()[0][env_ids]
        
        target = self.target_position.expand_as(drone_pos)
        return torch.norm(drone_pos - target, dim=-1)
    
    def _get_observations(self) -> TensorDict:
        """Get current observations."""
        drone_pos, drone_rot = self.drone.get_world_poses()
        drone_vel, drone_ang_vel = self.drone.get_velocities()
        
        relative_target = self.target_position - drone_pos
        distance = torch.norm(relative_target, dim=-1, keepdim=True)
        
        obs = torch.cat([
            drone_pos, drone_vel, drone_ang_vel, relative_target, distance
        ], dim=-1).unsqueeze(1)
        
        return TensorDict({
            "agents": TensorDict({
                "observation": obs,
            }, batch_size=[self.num_envs]),
            "stats": TensorDict({
                "return": self.episode_return.unsqueeze(-1),
                "episode_len": self.episode_length.unsqueeze(-1).float(),
                "success": self.success.unsqueeze(-1).float(),
            }, batch_size=[self.num_envs]),
        }, batch_size=[self.num_envs])
    
    def _compute_reward_and_done(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reward and termination."""
        distance = self._compute_distance()
        
        # Reward components
        reward = self.reward_cfg.get("distance_scale", -1.0) * distance
        reward += self.reward_cfg.get("progress_scale", 20.0) * (self.prev_distance - distance)
        reward += self.reward_cfg.get("alive_bonus", 0.1)
        
        # Success
        success = distance < self.target_threshold
        reward += success.float() * self.reward_cfg.get("success_bonus", 100.0)
        
        # Crash
        drone_pos = self.drone.get_world_poses()[0]
        crashed = drone_pos[:, 2] < 0.1
        reward += crashed.float() * self.reward_cfg.get("crash_penalty", -50.0)
        
        # Too far
        too_far = distance > self.max_distance
        reward += too_far.float() * self.reward_cfg.get("crash_penalty", -50.0)
        
        # Done
        done = success | crashed | too_far | (self.episode_length >= self.max_episode_length)
        
        # Update tracking
        self.prev_distance = distance.clone()
        self.episode_return += reward
        self.success = success.float()
        
        return reward.unsqueeze(-1), done.unsqueeze(-1)
    
    def _post_sim_step(self, tensordict: TensorDict) -> TensorDict:
        """Process after simulation step."""
        self.episode_length += 1
        reward, done = self._compute_reward_and_done()
        obs = self._get_observations()
        
        tensordict.update({
            "next": obs,
            "reward": reward,
            "done": done,
        })
        return tensordict


# Alias
DroneToTarget = DroneToTargetTask
