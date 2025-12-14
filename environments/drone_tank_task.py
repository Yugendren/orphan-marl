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
        self.camera_handles = []  # Store camera handles
        
        # State tracking
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Initialize environment
        self._create_envs()
        
        # Get observation and action spaces
        self._setup_observations()
        self._setup_actions()
        
        # Prepare tensor access
        self._prepare_tensors()
        
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
            
            self.camera_handles.append(camera_handle)
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
        camera_enabled = self.cfg.get("observations", {}).get("camera", {}).get("enabled", True)
        if camera_enabled:
            self.use_vision = True
            camera_cfg = self.cfg.get("drone", {}).get("sensors", {}).get("camera", {})
            self.camera_height = camera_cfg.get("resolution", [640, 480])[1]
            self.camera_width = camera_cfg.get("resolution", [640, 480])[0]
        else:
            self.use_vision = False
            self.camera_height = 0
            self.camera_width = 0
        
        # Total observation size
        self.obs_buf = torch.zeros((self.num_envs, self.num_state_obs), 
                                   device=self.device, dtype=torch.float)
        
        # Vision buffer for camera images
        if self.use_vision:
            self.vision_buf = torch.zeros((self.num_envs, 3, self.camera_height, self.camera_width), 
                                         device=self.device, dtype=torch.float)
        else:
            self.vision_buf = None
        
    def _setup_actions(self):
        """Setup action space."""
        self.num_actions = self.cfg.get("actions", {}).get("num_actions", 4)
        self.actions = torch.zeros((self.num_envs, self.num_actions), 
                                   device=self.device, dtype=torch.float)
    
    def _prepare_tensors(self):
        """Prepare tensors for efficient state access."""
        # Get rigid body tensor handles for efficient access
        actor_indices = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        
        # Get handles for all actors
        self.drone_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.tank_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        for i in range(self.num_envs):
            self.drone_indices[i] = self.gym.get_actor_index(self.envs[i], self.drone_handles[i], gymapi.DOMAIN_SIM)
            self.tank_indices[i] = self.gym.get_actor_index(self.envs[i], self.tank_handles[i], gymapi.DOMAIN_SIM)
    
    def compute_observations(self):
        """Compute observations from the environment."""
        # Extract state observations for all environments
        for i in range(self.num_envs):
            # Get actor rigid body states
            rb_states = self.gym.get_actor_rigid_body_states(
                self.envs[i], self.drone_handles[i], gymapi.STATE_ALL
            )
            
            # Position (3D) - from root body (index 0)
            pos = rb_states['pose']['p'][0]
            self.obs_buf[i, 0:3] = to_torch([pos.x, pos.y, pos.z], device=self.device)
            
            # Velocity (3D)
            vel = rb_states['vel']['linear'][0]
            self.obs_buf[i, 3:6] = to_torch([vel.x, vel.y, vel.z], device=self.device)
            
            # Orientation (quaternion, 4D)
            quat = rb_states['pose']['r'][0]
            self.obs_buf[i, 6:10] = to_torch([quat.x, quat.y, quat.z, quat.w], device=self.device)
            
            # Angular velocity (3D)
            ang_vel = rb_states['vel']['angular'][0]
            self.obs_buf[i, 10:13] = to_torch([ang_vel.x, ang_vel.y, ang_vel.z], device=self.device)
        
        # Get camera images if vision is enabled
        if self.use_vision and len(self.camera_handles) > 0:
            # Render all camera sensors
            self.gym.render_all_camera_sensors(self.sim)
            
            for i in range(self.num_envs):
                # Get camera image
                camera_image = self.gym.get_camera_image(
                    self.sim, self.envs[i], self.camera_handles[i], 
                    gymapi.IMAGE_COLOR
                )
                
                if camera_image is not None:
                    # Convert to tensor: (H, W, C) -> (C, H, W)
                    image_tensor = torch.from_numpy(camera_image).to(self.device)
                    if image_tensor.dim() == 3 and image_tensor.shape[2] == 3:
                        image_tensor = image_tensor.permute(2, 0, 1)
                    
                    # Normalize to [0, 1]
                    image_tensor = image_tensor.float() / 255.0
                    
                    # Resize if needed
                    if image_tensor.shape[1:] != (self.camera_height, self.camera_width):
                        import torch.nn.functional as F
                        image_tensor = F.interpolate(
                            image_tensor.unsqueeze(0),
                            size=(self.camera_height, self.camera_width),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    self.vision_buf[i] = image_tensor
        
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
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        
        for i in range(self.num_envs):
            rb_states = self.gym.get_actor_rigid_body_states(
                self.envs[i], self.drone_handles[i], gymapi.STATE_POS
            )
            pos = rb_states['pose']['p'][0]  # Root body position
            positions[i] = to_torch([pos.x, pos.y, pos.z], device=self.device)
        
        return positions
    
    def _get_tank_positions(self):
        """Get tank positions for all environments."""
        positions = torch.zeros((self.num_envs, 3), device=self.device)
        
        for i in range(self.num_envs):
            rb_states = self.gym.get_actor_rigid_body_states(
                self.envs[i], self.tank_handles[i], gymapi.STATE_POS
            )
            pos = rb_states['pose']['p'][0]  # Root body position
            positions[i] = to_torch([pos.x, pos.y, pos.z], device=self.device)
        
        return positions
    
    def reset_idx(self, env_ids):
        """Reset specific environments."""
        # Reset drone positions
        for env_id in env_ids:
            # Randomize drone spawn position
            drone_spawn = self.cfg.get("drone", {}).get("spawn", {})
            base_pos = drone_spawn.get("translation", [0.0, 5.0, 0.0])
            pos_x = base_pos[0] + np.random.uniform(-2.0, 2.0)
            pos_y = base_pos[1] + np.random.uniform(-2.0, 2.0)
            pos_z = base_pos[2] + np.random.uniform(0.0, 2.0)
            
            drone_pose = gymapi.Transform()
            drone_pose.p = gymapi.Vec3(pos_x, pos_y, pos_z)
            drone_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            # Reset drone pose using set_actor_root_state_tensor or individual set
            # Create rigid body state
            rb_states = self.gym.get_actor_rigid_body_states(
                self.envs[env_id], self.drone_handles[env_id], gymapi.STATE_ALL
            )
            rb_states['pose']['p'][0] = drone_pose.p
            rb_states['pose']['r'][0] = drone_pose.r
            rb_states['vel']['linear'][0] = gymapi.Vec3(0, 0, 0)
            rb_states['vel']['angular'][0] = gymapi.Vec3(0, 0, 0)
            
            self.gym.set_actor_rigid_body_states(
                self.envs[env_id], self.drone_handles[env_id], rb_states, gymapi.STATE_ALL
            )
            
            # Randomize tank position
            tank_spawn = self.cfg.get("tank", {}).get("spawn", {})
            base_tank_pos = tank_spawn.get("translation", [10.0, 0.0, 0.0])
            tank_x = base_tank_pos[0] + np.random.uniform(-5.0, 5.0)
            tank_y = base_tank_pos[1] + np.random.uniform(-5.0, 5.0)
            tank_z = base_tank_pos[2]
            
            tank_pose = gymapi.Transform()
            tank_pose.p = gymapi.Vec3(tank_x, tank_y, tank_z)
            tank_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            
            # Reset tank pose
            tank_rb_states = self.gym.get_actor_rigid_body_states(
                self.envs[env_id], self.tank_handles[env_id], gymapi.STATE_ALL
            )
            tank_rb_states['pose']['p'][0] = tank_pose.p
            tank_rb_states['pose']['r'][0] = tank_pose.r
            tank_rb_states['vel']['linear'][0] = gymapi.Vec3(0, 0, 0)
            tank_rb_states['vel']['angular'][0] = gymapi.Vec3(0, 0, 0)
            
            self.gym.set_actor_rigid_body_states(
                self.envs[env_id], self.tank_handles[env_id], tank_rb_states, gymapi.STATE_ALL
            )
    
    def pre_physics_step(self, actions):
        """Apply actions before physics step."""
        self.actions = actions.clone().to(self.device)
        
        # Apply actions to drone (thrust forces)
        # Actions are: [vx, vy, vz, yaw_rate] - convert to forces/torques
        # For simplicity, we'll apply forces in world frame
        # In a real implementation, you'd convert to body frame and apply proper drone dynamics
        
        # Get root body indices for all drones
        for i in range(self.num_envs):
            # Get root body handle
            root_body = self.gym.get_actor_rigid_body_dict(self.envs[i], self.drone_handles[i])['body']
            
            # Convert actions to forces (simplified - in practice use proper drone model)
            # Actions are velocity commands, convert to forces
            force_scale = 10.0
            torque_scale = 5.0
            
            force = gymapi.Vec3(
                self.actions[i, 0].item() * force_scale,
                self.actions[i, 1].item() * force_scale,
                self.actions[i, 2].item() * force_scale
            )
            torque = gymapi.Vec3(0.0, 0.0, self.actions[i, 3].item() * torque_scale)
            
            # Apply force and torque to root body
            self.gym.apply_rigid_body_force_at_pos(
                self.envs[i], self.drone_handles[i], root_body, force, 
                gymapi.Vec3(0.0, 0.0, 0.0), gymapi.ENV_SPACE
            )
            self.gym.apply_rigid_body_torque(
                self.envs[i], self.drone_handles[i], torque, gymapi.ENV_SPACE
            )
    
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

