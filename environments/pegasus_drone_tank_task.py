"""
Pegasus Simulator Environment for Drone Tank Task
Uses Pegasus Simulator API with PX4 integration
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.vehicles import Multirotor
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend
from pegasus.simulator.logic.graphs import Graph
from pegasus.simulator.logic.sensors import Camera
from pegasus.simulator.logic.interface import PegasusInterface


class PegasusDroneTankTask:
    """
    Drone Tank Task using Pegasus Simulator with PX4 integration.
    Uses camera vision to detect tank and navigate towards it.
    """
    
    def __init__(self, cfg: Dict[str, Any], device: str = "cuda:0"):
        """
        Initialize Pegasus environment.
        
        Args:
            cfg: Configuration dictionary
            device: Device for tensors
        """
        self.cfg = cfg
        self.device = torch.device(device)
        
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
        
        # Contact thresholds
        self.tank_contact_threshold = self.reward_cfg.get("tank_contact_threshold", 0.5)
        self.ground_contact_threshold = self.reward_cfg.get("ground_contact_threshold", 0.3)
        
        # Initialize Pegasus
        self.world = World(stage_units_in_meters=1.0)
        self.interface = PegasusInterface()
        
        # Vehicle and sensor handles
        self.vehicle = None
        self.camera = None
        self.tank_prim = None
        
        # State tracking
        self.progress_buf = 0
        self.reset_buf = False
        
        # Observation and action spaces
        self._setup_observations()
        self._setup_actions()
        
        # Initialize environment
        self._create_env()
    
    def _setup_observations(self):
        """Setup observation space."""
        # State observations: position, velocity, orientation, angular velocity
        self.num_state_obs = 13  # 3 pos + 3 vel + 4 quat + 3 ang_vel
        
        # Vision observations
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
        
        # Observation buffers
        self.obs_buf = torch.zeros(self.num_state_obs, device=self.device, dtype=torch.float)
        if self.use_vision:
            self.vision_buf = torch.zeros((3, self.camera_height, self.camera_width), 
                                         device=self.device, dtype=torch.float)
        else:
            self.vision_buf = None
    
    def _setup_actions(self):
        """Setup action space."""
        self.num_actions = self.cfg.get("actions", {}).get("num_actions", 4)
        self.action_scale = torch.tensor(
            self.cfg.get("actions", {}).get("scale", [1.0] * self.num_actions),
            device=self.device
        )
    
    def _create_env(self):
        """Create Pegasus simulation environment."""
        # Load tank USD asset
        tank_cfg = self.cfg.get("tank", {})
        tank_usd_path = tank_cfg.get("usd_path", None)
        
        if tank_usd_path:
            # Load tank from USD
            self.tank_prim = self.world.scene.add(
                stage_utils.add_reference_to_stage(tank_usd_path, "/tank")
            )
        else:
            # Create placeholder tank (box)
            from omni.isaac.core.prims import XFormPrim
            self.tank_prim = self.world.scene.add(
                XFormPrim(prim_path="/tank", name="tank", position=np.array([10.0, 0.0, 0.0]))
            )
        
        # Create vehicle using Pegasus
        vehicle_cfg = self.cfg.get("drone", {})
        vehicle_usd_path = vehicle_cfg.get("usd_path", "/home/ubuntu/isaac_assets/SM_Tank_T72b.usd")
        
        # Initialize vehicle with PX4 backend
        spawn_pos = vehicle_cfg.get("spawn", {}).get("translation", [0.0, 5.0, 0.0])
        
        # Create Multirotor vehicle
        self.vehicle = Multirotor(
            model="iris",  # gazebo-classic_iris airframe
            id=0,
            spawn_position=spawn_pos,
            spawn_orientation=[0.0, 0.0, 0.0, 1.0],
            usd_path=vehicle_usd_path
        )
        
        # Add camera sensor if enabled
        if self.use_vision:
            camera_cfg = vehicle_cfg.get("sensors", {}).get("camera", {})
            camera_resolution = camera_cfg.get("resolution", [640, 480])
            camera_fov = camera_cfg.get("fov", 60.0)
            camera_pos = camera_cfg.get("position", [0.0, 0.0, 0.0])
            camera_orient = camera_cfg.get("orientation", [0.0, 0.0, 0.0, 1.0])
            
            self.camera = Camera(
                name="camera",
                vehicle=self.vehicle,
                resolution=camera_resolution,
                fov=camera_fov,
                position=camera_pos,
                orientation=camera_orient
            )
            
            # Add camera to vehicle
            self.vehicle.add_sensor(self.camera)
        
        # Setup PX4 backend
        backend = MavlinkBackend(
            vehicle=self.vehicle,
            connection_string="udp://:14540",  # PX4 connection
            vehicle_id=0
        )
        
        # Add vehicle to world
        self.world.scene.add(self.vehicle)
        
        # Initialize world
        self.world.reset()
    
    def compute_observations(self):
        """Compute observations from environment."""
        # Get vehicle state
        state = self.vehicle.get_state()
        
        # Extract state observations
        position = torch.tensor([state.position.x, state.position.y, state.position.z], 
                                device=self.device)
        velocity = torch.tensor([state.velocity.x, state.velocity.y, state.velocity.z], 
                               device=self.device)
        orientation = torch.tensor([state.orientation.x, state.orientation.y, 
                                   state.orientation.z, state.orientation.w], 
                                  device=self.device)
        angular_velocity = torch.tensor([state.angular_velocity.x, 
                                       state.angular_velocity.y, 
                                       state.angular_velocity.z], 
                                      device=self.device)
        
        # Combine state observations
        self.obs_buf = torch.cat([position, velocity, orientation, angular_velocity])
        
        # Get camera images if vision is enabled
        if self.use_vision and self.camera is not None:
            # Get camera image from Pegasus
            camera_image = self.camera.get_data()
            
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
                
                self.vision_buf = image_tensor
        
        return self.obs_buf
    
    def compute_rewards(self):
        """Compute rewards based on task performance."""
        # Get vehicle and tank positions
        vehicle_state = self.vehicle.get_state()
        drone_pos = torch.tensor([vehicle_state.position.x, 
                                  vehicle_state.position.y, 
                                  vehicle_state.position.z], 
                                 device=self.device)
        
        # Get tank position (from prim)
        tank_world_pose = self.tank_prim.get_world_pose()
        tank_pos = torch.tensor([tank_world_pose[0][0], 
                                 tank_world_pose[0][1], 
                                 tank_world_pose[0][2]], 
                                device=self.device)
        
        # Calculate distances
        distance_to_tank = torch.norm(drone_pos - tank_pos)
        distance_to_ground = torch.abs(drone_pos[2] - 0.0)
        
        # Tank hit reward
        tank_hit = (distance_to_tank < self.tank_contact_threshold).float()
        tank_reward = tank_hit * self.tank_hit_reward
        
        # Ground crash penalty
        ground_crash = (distance_to_ground < self.ground_contact_threshold).float()
        crash_penalty = ground_crash * self.ground_crash_penalty
        
        # Distance reward
        distance_reward = -distance_to_tank * self.distance_reward_scale
        
        # Survival reward
        survival_reward = torch.tensor(self.survival_reward, device=self.device)
        
        # Total reward
        reward = tank_reward + crash_penalty + distance_reward + survival_reward
        
        # Update reset buffer
        if self.reset_on_crash:
            self.reset_buf = self.reset_buf or (ground_crash.item() > 0)
        if self.reset_on_success:
            self.reset_buf = self.reset_buf or (tank_hit.item() > 0)
        
        return reward.item()
    
    def reset(self):
        """Reset environment."""
        # Reset vehicle position
        spawn_cfg = self.cfg.get("drone", {}).get("spawn", {})
        spawn_pos = spawn_cfg.get("translation", [0.0, 5.0, 0.0])
        self.vehicle.set_state(position=spawn_pos, orientation=[0.0, 0.0, 0.0, 1.0])
        
        # Randomize tank position
        tank_cfg = self.cfg.get("tank", {})
        tank_spawn = tank_cfg.get("spawn", {}).get("translation", [10.0, 0.0, 0.0])
        tank_x = np.random.uniform(5.0, 15.0)
        tank_y = np.random.uniform(-5.0, 5.0)
        self.tank_prim.set_world_pose(position=np.array([tank_x, tank_y, 0.0]))
        
        # Reset buffers
        self.progress_buf = 0
        self.reset_buf = False
        
        # Compute initial observations
        obs = self.compute_observations()
        return obs
    
    def step(self, actions: torch.Tensor):
        """
        Step environment.
        
        Args:
            actions: Action tensor [num_actions]
            
        Returns:
            obs: Observations
            reward: Reward scalar
            done: Done flag
            info: Info dict
        """
        # Scale actions
        scaled_actions = actions * self.action_scale
        
        # Send actions to PX4 via offboard mode
        # Convert actions to velocity/position setpoints for PX4
        # For PX4, we typically use velocity_ned or position_ned commands
        vx, vy, vz, yaw_rate = scaled_actions[0], scaled_actions[1], scaled_actions[2], scaled_actions[3]
        
        # Send velocity command to PX4 (via MAVSDK or direct PX4 interface)
        # This would use the MavlinkBackend to send offboard commands
        # For now, we'll use the vehicle's control interface
        self.vehicle.set_velocity_command([vx, vy, vz], yaw_rate)
        
        # Step simulation
        self.world.step(render=False)
        
        # Compute observations
        obs = self.compute_observations()
        
        # Compute rewards
        reward = self.compute_rewards()
        
        # Update progress
        self.progress_buf += 1
        done = self.reset_buf or (self.progress_buf >= self.max_episode_length)
        
        info = {
            "tank_hit": self.reset_buf and self.reset_on_success,
            "crashed": self.reset_buf and self.reset_on_crash,
        }
        
        if done:
            obs = self.reset()
        
        return obs, reward, done, info
