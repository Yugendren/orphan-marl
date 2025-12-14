"""
Drone Tank Task Environment for Isaac Sim/Lab
Drone learns to identify and strike a tank using CV information
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

from isaacgym import gymapi, gymutil
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params


class DroneTankTask(VecTask):
    """
    Task where a drone must identify and strike a tank.
    Uses camera vision to detect the tank and navigate towards it.
    """
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """
        Initialize the Drone Tank Task environment.
        
        Args:
            cfg: Configuration dictionary
            rl_device: Device for RL computations
            sim_device: Device for simulation
            graphics_device_id: Graphics device ID
            headless: Whether to run headless
            virtual_screen_capture: Virtual screen capture setting
            force_render: Force render setting
        """
        self.cfg = cfg
        
        # Reward parameters
        self.reward_cfg = cfg.get("reward", {})
        self.tank_hit_reward = self.reward_cfg.get("tank_hit", 10.0)
        self.ground_crash_penalty = self.reward_cfg.get("ground_crash", -5.0)
        self.distance_reward_scale = self.reward_cfg.get("distance_scale", 0.1)
        self.survival_reward = self.reward_cfg.get("survival", 0.01)
        
        # Task parameters
        self.max_episode_length = cfg.get("episode", {}).get("max_length", 1000)
        self.reset_on_crash = cfg.get("episode", {}).get("reset_on_crash", True)
        self.reset_on_success = cfg.get("episode", {}).get("reset_on_success", True)
        
        # Contact detection thresholds
        self.tank_contact_threshold = 0.5  # Distance threshold for tank contact
        self.ground_contact_threshold = 0.3  # Distance threshold for ground contact
        
        # Initialize base class
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        # Get asset handles
        self.drone_handles = []
        self.tank_handles = []
        self.ground_handles = []
        
        # State tracking
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Initialize environment
        self._create_envs()
        
        # Get observation and action spaces
        self._setup_observations()
        self._setup_actions()
        
    def _create_envs(self):
        """Create simulation environments."""
        # Load assets
        drone_asset = self._load_drone_asset()
        tank_asset = self._load_tank_asset()
        
        # Create environments
        env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
        env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
        
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            
            # Spawn drone
            drone_pose = gymapi.Transform()
            drone_pose.p = gymapi.Vec3(0.0, 5.0, 0.0)
            drone_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            drone_handle = self.gym.create_actor(env, drone_asset, drone_pose, "drone", i, 1)
            self.drone_handles.append(drone_handle)
            
            # Spawn tank at random position
            tank_pose = gymapi.Transform()
            tank_x = np.random.uniform(5.0, 15.0)
            tank_y = np.random.uniform(-5.0, 5.0)
            tank_pose.p = gymapi.Vec3(tank_x, tank_y, 0.0)
            tank_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            tank_handle = self.gym.create_actor(env, tank_asset, tank_pose, "tank", i, 2)
            self.tank_handles.append(tank_handle)
            
            # Add camera sensor to drone
            camera_props = gymapi.CameraProperties()
            camera_props.width = 640
            camera_props.height = 480
            camera_props.horizontal_fov = 60.0
            
            camera_handle = self.gym.create_camera_sensor(env, camera_props)
            camera_offset = gymapi.Vec3(0.0, 0.0, 0.0)
            camera_rotation = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            self.gym.attach_camera_to_body(camera_handle, env, drone_handle, 
                                          gymapi.Transform(camera_offset, camera_rotation),
                                          gymapi.FOLLOW_TRANSFORM)
            
            self.envs.append(env)
    
    def _load_drone_asset(self):
        """Load drone USD asset."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.02
        
        # If USD path is provided, load it; otherwise create a simple box
        drone_cfg = self.cfg.get("drone", {})
        usd_path = drone_cfg.get("usd_path", None)
        
        if usd_path:
            asset = self.gym.load_asset(self.sim, "", usd_path, asset_options)
        else:
            # Create a simple box as placeholder
            asset = self.gym.create_box(self.sim, 0.5, 0.5, 0.2, asset_options)
        
        return asset
    
    def _load_tank_asset(self):
        """Load tank USD asset."""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        
        tank_cfg = self.cfg.get("tank", {})
        usd_path = tank_cfg.get("usd_path", None)
        
        if usd_path:
            asset = self.gym.load_asset(self.sim, "", usd_path, asset_options)
        else:
            # Create a simple box as placeholder
            asset = self.gym.create_box(self.sim, 2.0, 2.0, 1.0, asset_options)
        
        return asset
    
    def _setup_observations(self):
        """Setup observation space."""
        # State observations: position, velocity, orientation, angular velocity
        self.num_state_obs = 13  # 3 pos + 3 vel + 4 quat + 3 ang_vel
        
        # Vision observations: camera image
        self.num_vision_obs = 640 * 480 * 3  # RGB image
        
        # Total observation size
        self.obs_buf = torch.zeros((self.num_envs, self.num_state_obs), 
                                   device=self.device, dtype=torch.float)
        
    def _setup_actions(self):
        """Setup action space."""
        self.num_actions = self.cfg.get("actions", {}).get("num_actions", 4)
        self.actions = torch.zeros((self.num_envs, self.num_actions), 
                                   device=self.device, dtype=torch.float)
    
    def compute_observations(self):
        """Compute observations from the environment."""
        # Get drone states
        drone_states = self.gym.get_actor_rigid_body_states(self.envs[0], 
                                                           self.drone_handles[0], 
                                                           gymapi.STATE_ALL)
        
        # Extract position, velocity, orientation, angular velocity
        # This is a simplified version - in practice, you'd need to handle all envs
        # For now, we'll use state-based observations
        
        # Get camera images
        # camera_images = self.gym.get_camera_image(...)
        
        # Combine observations
        # self.obs_buf = torch.cat([state_obs, vision_obs], dim=-1)
        
        # Placeholder: return state observations only for now
        return self.obs_buf
    
    def compute_rewards(self):
        """Compute rewards based on task performance."""
        # Get drone and tank positions
        drone_pos = self._get_drone_positions()
        tank_pos = self._get_tank_positions()
        ground_height = 0.0
        
        # Calculate distances
        distance_to_tank = torch.norm(drone_pos - tank_pos, dim=-1)
        distance_to_ground = torch.abs(drone_pos[:, 2] - ground_height)
        
        # Tank hit reward (positive when close to tank)
        tank_hit = (distance_to_tank < self.tank_contact_threshold).float()
        tank_reward = tank_hit * self.tank_hit_reward
        
        # Ground crash penalty (negative when too close to ground)
        ground_crash = (distance_to_ground < self.ground_contact_threshold).float()
        crash_penalty = ground_crash * self.ground_crash_penalty
        
        # Distance-based reward (encourage getting closer to tank)
        distance_reward = -distance_to_tank * self.distance_reward_scale
        
        # Survival reward (small positive reward for staying alive)
        survival_reward = torch.ones_like(distance_to_tank) * self.survival_reward
        
        # Total reward
        rewards = tank_reward + crash_penalty + distance_reward + survival_reward
        
        # Update reset buffer
        if self.reset_on_crash:
            self.reset_buf = torch.where(ground_crash.bool(), 
                                        torch.ones_like(self.reset_buf), 
                                        self.reset_buf)
        
        if self.reset_on_success:
            self.reset_buf = torch.where(tank_hit.bool(), 
                                        torch.ones_like(self.reset_buf), 
                                        self.reset_buf)
        
        return rewards
    
    def _get_drone_positions(self):
        """Get drone positions for all environments."""
        # Placeholder - in practice, get from gym API
        return torch.zeros((self.num_envs, 3), device=self.device)
    
    def _get_tank_positions(self):
        """Get tank positions for all environments."""
        # Placeholder - in practice, get from gym API
        return torch.ones((self.num_envs, 3), device=self.device) * 10.0
    
    def reset_idx(self, env_ids):
        """Reset specific environments."""
        # Reset drone and tank positions
        # In practice, randomize spawn positions
        pass
    
    def pre_physics_step(self, actions):
        """Apply actions before physics step."""
        self.actions = actions.clone().to(self.device)
        
        # Apply actions to drone (thrust forces)
        # In practice, convert actions to forces/torques
        pass
    
    def post_physics_step(self):
        """Compute observations and rewards after physics step."""
        self.progress_buf += 1
        
        # Compute observations
        self.obs_buf = self.compute_observations()
        
        # Compute rewards
        self.rew_buf = self.compute_rewards()
        
        # Reset environments that are done
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        # Reset progress buffer for reset environments
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

