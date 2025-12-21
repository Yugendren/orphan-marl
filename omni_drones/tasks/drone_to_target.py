"""
DroneToTarget Task for OmniDrones

A custom RL task where a quadrotor drone must navigate to a randomly placed target position.

Observation Space:
    - Drone position [3]
    - Drone linear velocity [3]
    - Drone angular velocity [3]
    - Relative target position [3]
    - Distance to target [1]
    Total: 13-dimensional state vector

Action Space:
    - Velocity commands [vx, vy, vz] or motor commands depending on config

Reward:
    - Distance-based penalty
    - Progress reward for getting closer
    - Success bonus for reaching target
    - Crash penalty

Episode Termination:
    - Drone reaches target (success)
    - Drone crashes (ground collision)
    - Max steps reached (truncation)
"""

import torch
import numpy as np
from typing import Dict, Optional

from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView

from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec

# Register this task with OmniDrones
from omni_drones.envs import register_env


@register_env("DroneToTarget")
class DroneToTargetTask(IsaacEnv):
    """
    Navigate drone to randomly placed target position.
    
    GPU-parallelized environment supporting 1000s of simultaneous environments.
    """
    
    def __init__(self, cfg, headless: bool):
        """
        Initialize the DroneToTarget task.
        
        Args:
            cfg: Hydra configuration
            headless: Whether to run without GUI
        """
        self.target_threshold = cfg.task.target.get("distance_threshold", 0.5)
        self.max_distance = cfg.task.target.get("max_distance", 10.0)
        self.height_range = cfg.task.target.get("height_range", [1.0, 3.0])
        self.randomize_target = cfg.task.target.get("randomize", True)
        
        # Reward scales
        self.reward_cfg = cfg.task.reward
        
        super().__init__(cfg, headless)
        
        # Previous distance for progress calculation
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)
        
    def _design_scene(self):
        """
        Design the simulation scene.
        
        Spawns:
            - Ground plane
            - Drone (Iris quadrotor by default)
            - Target marker (visual only)
        """
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import UsdGeom, Gf
        
        # Spawn ground plane
        prim_utils.create_prim("/World/ground", "Plane")
        
        # Spawn drone
        self.drone = MultirotorBase(
            name="drone",
            drone_model=self.cfg.task.drone.get("model", "Iris"),
            spawn_pos=torch.tensor([0.0, 0.0, 1.5], device=self.device).expand(self.num_envs, 3),
        )
        
        # Create visual target markers
        for i in range(min(self.num_envs, 10)):  # Only visual for first 10 envs
            sphere_path = f"/World/env_{i}/target_sphere"
            sphere = UsdGeom.Sphere.Define(self._stage, sphere_path)
            sphere.GetRadiusAttr().Set(0.1)
            sphere.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])  # Red
    
    def _set_specs(self):
        """
        Define observation and action specifications.
        """
        obs_dim = 13  # pos(3) + vel(3) + ang_vel(3) + rel_target(3) + dist(1)
        
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    shape=(self.num_envs, 1, obs_dim),  # (env, agent, obs)
                    device=self.device
                ),
            }),
            "stats": CompositeSpec({
                "return": UnboundedContinuousTensorSpec((self.num_envs, 1)),
                "episode_len": UnboundedContinuousTensorSpec((self.num_envs, 1)),
                "success": UnboundedContinuousTensorSpec((self.num_envs, 1)),
            }),
        })
        
        # Action space: velocity commands [vx, vy, vz]
        action_dim = 3
        max_vel = self.cfg.task.action.get("max_velocity", [2.0, 2.0, 1.0])
        
        self.action_spec = BoundedTensorSpec(
            low=torch.tensor([-max_vel[0], -max_vel[1], -max_vel[2]]),
            high=torch.tensor([max_vel[0], max_vel[1], max_vel[2]]),
            shape=(self.num_envs, 1, action_dim),
            device=self.device
        )
        
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(self.num_envs, 1),
            device=self.device
        )
        
        self.done_spec = BoundedTensorSpec(
            low=0,
            high=1,
            shape=(self.num_envs, 1),
            dtype=torch.bool,
            device=self.device
        )
    
    def _reset_idx(self, env_ids: torch.Tensor):
        """
        Reset specified environments.
        
        Args:
            env_ids: Tensor of environment indices to reset
        """
        n = len(env_ids)
        
        # Reset drone position
        spawn_pos = torch.zeros(n, 3, device=self.device)
        spawn_pos[:, 2] = self.cfg.task.drone.get("spawn_pos", [0, 0, 1.5])[2]
        self.drone.set_world_poses(spawn_pos, env_ids=env_ids)
        
        # Reset drone velocity
        zeros = torch.zeros(n, 3, device=self.device)
        self.drone.set_velocities(zeros, zeros, env_ids=env_ids)
        
        # Randomize target positions
        if self.randomize_target:
            self.target_positions[env_ids, 0] = torch.rand(n, device=self.device) * 8.0 - 4.0
            self.target_positions[env_ids, 1] = torch.rand(n, device=self.device) * 8.0 - 4.0
            self.target_positions[env_ids, 2] = torch.rand(n, device=self.device) * (
                self.height_range[1] - self.height_range[0]
            ) + self.height_range[0]
        
        # Reset tracking variables
        self.prev_distance[env_ids] = self._compute_distance(env_ids)
        self.episode_length[env_ids] = 0
        self.episode_return[env_ids] = 0
        self.success[env_ids] = 0
    
    def _pre_sim_step(self, tensordict: TensorDict):
        """
        Apply actions before simulation step.
        
        Args:
            tensordict: TensorDict containing action
        """
        # Get velocity commands
        actions = tensordict["agents"]["action"].squeeze(1)  # (num_envs, 3)
        
        # Apply velocity controller
        self.drone.apply_velocity_commands(actions)
    
    def _compute_distance(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute distance from drone to target."""
        if env_ids is None:
            drone_pos = self.drone.get_world_poses()[0]
            target_pos = self.target_positions
        else:
            drone_pos = self.drone.get_world_poses()[0][env_ids]
            target_pos = self.target_positions[env_ids]
        
        return torch.norm(drone_pos - target_pos, dim=-1)
    
    def _get_observations(self) -> TensorDict:
        """
        Compute observations for all environments.
        
        Returns:
            TensorDict with observation data
        """
        # Get drone state
        drone_pos, drone_rot = self.drone.get_world_poses()
        drone_vel, drone_ang_vel = self.drone.get_velocities()
        
        # Relative target position
        relative_target = self.target_positions - drone_pos
        
        # Distance
        distance = torch.norm(relative_target, dim=-1, keepdim=True)
        
        # Concatenate observation
        obs = torch.cat([
            drone_pos,           # (num_envs, 3)
            drone_vel,           # (num_envs, 3)
            drone_ang_vel,       # (num_envs, 3)
            relative_target,     # (num_envs, 3)
            distance,            # (num_envs, 1)
        ], dim=-1)
        
        # Add agent dimension for MARL compatibility
        obs = obs.unsqueeze(1)  # (num_envs, 1, obs_dim)
        
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
    
    def _compute_reward_and_done(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reward and done flags.
        
        Returns:
            Tuple of (reward, done) tensors
        """
        # Get current distance
        distance = self._compute_distance()
        
        # Distance penalty
        reward = self.reward_cfg.distance_scale * distance
        
        # Progress reward
        progress = self.prev_distance - distance
        reward += self.reward_cfg.progress_scale * progress
        
        # Alive bonus
        reward += self.reward_cfg.get("alive_bonus", 0.1)
        
        # Check success
        success = distance < self.target_threshold
        reward += success.float() * self.reward_cfg.success_bonus
        
        # Check crash (ground collision)
        drone_pos = self.drone.get_world_poses()[0]
        crashed = drone_pos[:, 2] < 0.1
        reward += crashed.float() * self.reward_cfg.crash_penalty
        
        # Check too far
        too_far = distance > self.max_distance
        reward += too_far.float() * self.reward_cfg.crash_penalty
        
        # Done conditions
        done = success | crashed | too_far | (self.episode_length >= self.max_episode_length)
        
        # Update tracking
        self.prev_distance = distance.clone()
        self.episode_return += reward
        self.success = success.float()
        
        # Add agent dimension
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        
        return reward, done
    
    def _post_sim_step(self, tensordict: TensorDict) -> TensorDict:
        """
        Process after simulation step.
        
        Args:
            tensordict: Input TensorDict
            
        Returns:
            Updated TensorDict with observations, rewards, done
        """
        self.episode_length += 1
        
        reward, done = self._compute_reward_and_done()
        obs = self._get_observations()
        
        tensordict.update({
            "next": obs,
            "reward": reward,
            "done": done,
        })
        
        return tensordict


# Alias for registration
DroneToTarget = DroneToTargetTask
