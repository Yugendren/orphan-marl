"""
Drone Tank Task Environment for Isaac Lab
Migrated from Isaac Gym to Isaac Lab APIs
Drone learns to identify and strike a tank using CV information
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, wrap_to_pi


@configclass
class DroneCfg(RigidObjectCfg):
    """Configuration for the drone (using RigidObject since it's force-controlled)."""
    
    spawn = sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/isaac_assets/SM_Tank_T72b.usd",
        activate_contact_sensors=False,
        # Note: rigid_props will only work if USD file has RigidBodyAPI
        # If not, we'll need to apply it manually
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 5.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    )


@configclass
class TankCfg(RigidObjectCfg):
    """Configuration for the tank target."""
    
    spawn = sim_utils.UsdFileCfg(
        usd_path="/home/ubuntu/isaac_assets/SM_Tank_T72b.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            max_depenetration_velocity=5.0,
        ),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(10.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
    )


@configclass
class DroneTankTaskCfg(DirectRLEnvCfg):
    """Configuration for the Drone Tank Task environment."""
    
    # Environment settings
    decimation = 2  # env step every 2 sim steps
    episode_length_s = 20.0  # 20 seconds per episode
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=0.02,  # 50Hz simulation
        render_interval=decimation,
        device="cuda:0",  # Use GPU
    )
    
    # Scene settings
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=5.0,
        replicate_physics=True,
    )
    
    # Drone configuration
    drone_cfg: DroneCfg = DroneCfg(prim_path="/World/envs/env_.*/Drone")
    
    # Tank configuration
    tank_cfg: TankCfg = TankCfg(prim_path="/World/envs/env_.*/Tank")
    
    # Camera configuration (attached to drone)
    camera_cfg: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Drone/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="local",  # Relative to drone
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        width=640,
        height=480,
    )
    
    # Action and observation spaces
    action_space = 4  # [vx, vy, vz, yaw_rate]
    state_space = 13  # [pos(3), vel(3), quat(4), ang_vel(3)]
    observation_space = 13  # State observations: [pos(3), vel(3), quat(4), ang_vel(3)]
    
    # Action scaling
    action_scale = torch.tensor([1.0, 1.0, 1.0, 0.5])  # Scale for each action dimension
    
    # Reward parameters
    rew_scale_tank_hit = 10.0
    rew_scale_ground_crash = -5.0
    rew_scale_distance = 0.1
    rew_scale_survival = 0.01
    
    # Contact thresholds
    tank_contact_threshold = 0.5  # meters
    ground_contact_threshold = 0.3  # meters
    
    # Reset parameters
    drone_spawn_range = {
        "x": (-2.0, 2.0),
        "y": (-2.0, 2.0),
        "z": (0.0, 2.0),
    }
    tank_spawn_range = {
        "x": (5.0, 15.0),
        "y": (-5.0, 5.0),
        "z": (0.0, 0.0),
    }
    
    # Viewer settings
    viewer = ViewerCfg(eye=(10.0, 10.0, 10.0), lookat=(0.0, 0.0, 0.0))


class DroneTankTask(DirectRLEnv):
    """Drone Tank Task environment using Isaac Lab."""
    
    cfg: DroneTankTaskCfg
    
    def __init__(self, cfg: DroneTankTaskCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Store action scale
        self.action_scale = self.cfg.action_scale.to(self.device)
        
        # Vision support
        self.use_vision = "rgb" in self.cfg.camera_cfg.data_types
        
    def _setup_scene(self):
        """Setup the scene with drone, tank, and camera."""
        # Apply RigidBodyAPI to USD prims before creating RigidObject
        from pxr import UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage
        import isaaclab.sim.utils.prims as prim_utils
        
        # Spawn USD files first to get prims
        drone_prim_path = "/World/envs/env_0/Drone"
        tank_prim_path = "/World/envs/env_0/Tank"
        
        # Spawn the USD files
        self.cfg.drone_cfg.spawn.func(drone_prim_path, self.cfg.drone_cfg.spawn)
        self.cfg.tank_cfg.spawn.func(tank_prim_path, self.cfg.tank_cfg.spawn)
        
        # Apply RigidBodyAPI if not present
        stage = get_current_stage()
        for prim_path in [drone_prim_path, tank_prim_path]:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and not UsdPhysics.RigidBodyAPI(prim):
                UsdPhysics.RigidBodyAPI.Apply(prim)
        
        # Now create RigidObject instances
        self._drone = RigidObject(self.cfg.drone_cfg)
        self._tank = RigidObject(self.cfg.tank_cfg)
        self._camera = TiledCamera(self.cfg.camera_cfg)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        
        # Add to scene
        self.scene.rigid_objects["drone"] = self._drone
        self.scene.rigid_objects["tank"] = self._tank
        self.scene.sensors["camera"] = self._camera
        
        # Add ground plane
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg
        ground_cfg = GroundPlaneCfg(size=(100.0, 100.0))
        ground_cfg.func("/World/groundPlane", ground_cfg)
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Apply actions before physics step."""
        # Store actions for use in _apply_action
        self.actions = actions.clone()
    
    def _apply_action(self) -> None:
        """Apply actions to the drone."""
        # Scale actions
        scaled_actions = self.action_scale * self.actions
        
        # Convert actions to forces/torques
        # Actions: [vx, vy, vz, yaw_rate]
        # For simplicity, apply forces directly (in practice, use proper drone dynamics)
        force_scale = 10.0
        torque_scale = 5.0
        
        # Apply forces (simplified - in practice convert to body frame)
        forces = torch.zeros((self.num_envs, 3), device=self.device)
        forces[:, 0] = scaled_actions[:, 0] * force_scale  # x force
        forces[:, 1] = scaled_actions[:, 1] * force_scale  # y force
        forces[:, 2] = scaled_actions[:, 2] * force_scale  # z force
        
        torques = torch.zeros((self.num_envs, 3), device=self.device)
        torques[:, 2] = scaled_actions[:, 3] * torque_scale  # yaw torque
        
        # Apply forces and torques to root body using Isaac Lab API
        # For RigidObject, apply forces directly
        self._drone.root_physx_view.apply_forces_and_torques_at_position(
            forces=forces,
            torques=torques,
            positions=self._drone.data.root_pos_w,
        )
    
    def _get_observations(self) -> dict:
        """Get observations from the environment."""
        # Get drone state
        drone_pos = self._drone.data.root_pos_w
        drone_vel = self._drone.data.root_lin_vel_w
        drone_quat = self._drone.data.root_quat_w
        drone_ang_vel = self._drone.data.root_ang_vel_w
        
        # Combine state observations: [pos(3), vel(3), quat(4), ang_vel(3)] = 13D
        state_obs = torch.cat([
            drone_pos,
            drone_vel,
            drone_quat,
            drone_ang_vel,
        ], dim=-1)
        
        # For now, return state only (vision can be added later)
        # The observation space is set to 13D in config
        observations = {"policy": state_obs}
        
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Get positions
        drone_pos = self._drone.data.root_pos_w
        tank_pos = self._tank.data.root_pos_w
        
        # Calculate distances
        distance_to_tank = torch.norm(drone_pos - tank_pos, dim=-1)
        distance_to_ground = torch.abs(drone_pos[:, 2] - 0.0)
        
        # Tank hit reward
        tank_hit = (distance_to_tank < self.cfg.tank_contact_threshold).float()
        rew_tank_hit = self.cfg.rew_scale_tank_hit * tank_hit
        
        # Ground crash penalty
        ground_crash = (distance_to_ground < self.cfg.ground_contact_threshold).float()
        rew_ground_crash = self.cfg.rew_scale_ground_crash * ground_crash
        
        # Distance reward (encourage getting closer)
        rew_distance = -self.cfg.rew_scale_distance * distance_to_tank
        
        # Survival reward
        rew_survival = self.cfg.rew_scale_survival * torch.ones_like(distance_to_tank)
        
        # Total reward
        total_reward = rew_tank_hit + rew_ground_crash + rew_distance + rew_survival
        
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get done flags."""
        # Get positions
        drone_pos = self._drone.data.root_pos_w
        tank_pos = self._tank.data.root_pos_w
        
        # Calculate distances
        distance_to_tank = torch.norm(drone_pos - tank_pos, dim=-1)
        distance_to_ground = torch.abs(drone_pos[:, 2] - 0.0)
        
        # Termination conditions
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        tank_hit = distance_to_tank < self.cfg.tank_contact_threshold
        ground_crash = distance_to_ground < self.cfg.ground_contact_threshold
        
        # Terminate on tank hit or ground crash
        terminated = tank_hit | ground_crash
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset specific environments."""
        if env_ids is None:
            env_ids = self._drone._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        # Randomize drone spawn position
        drone_spawn_range = self.cfg.drone_spawn_range
        drone_pos = torch.zeros((len(env_ids), 3), device=self.device)
        drone_pos[:, 0] = sample_uniform(
            drone_spawn_range["x"][0],
            drone_spawn_range["x"][1],
            (len(env_ids),),
            self.device,
        )
        drone_pos[:, 1] = sample_uniform(
            drone_spawn_range["y"][0],
            drone_spawn_range["y"][1],
            (len(env_ids),),
            self.device,
        )
        drone_pos[:, 2] = sample_uniform(
            drone_spawn_range["z"][0],
            drone_spawn_range["z"][1],
            (len(env_ids),),
            self.device,
        )
        
        # Randomize tank spawn position
        tank_spawn_range = self.cfg.tank_spawn_range
        tank_pos = torch.zeros((len(env_ids), 3), device=self.device)
        tank_pos[:, 0] = sample_uniform(
            tank_spawn_range["x"][0],
            tank_spawn_range["x"][1],
            (len(env_ids),),
            self.device,
        )
        tank_pos[:, 1] = sample_uniform(
            tank_spawn_range["y"][0],
            tank_spawn_range["y"][1],
            (len(env_ids),),
            self.device,
        )
        tank_pos[:, 2] = torch.zeros((len(env_ids),), device=self.device)
        
        # Set drone state
        drone_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        drone_vel = torch.zeros((len(env_ids), 3), device=self.device)
        drone_ang_vel = torch.zeros((len(env_ids), 3), device=self.device)
        
        # Write to simulation
        drone_pos_world = drone_pos + self.scene.env_origins[env_ids]
        self._drone.write_root_pose_to_sim(torch.cat([drone_pos_world, drone_quat], dim=-1), env_ids)
        self._drone.write_root_velocity_to_sim(torch.cat([drone_vel, drone_ang_vel], dim=-1), env_ids)
        
        # Set tank state
        tank_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1)
        tank_vel = torch.zeros((len(env_ids), 3), device=self.device)
        tank_ang_vel = torch.zeros((len(env_ids), 3), device=self.device)
        
        tank_pos_world = tank_pos + self.scene.env_origins[env_ids]
        self._tank.write_root_pose_to_sim(torch.cat([tank_pos_world, tank_quat], dim=-1), env_ids)
        self._tank.write_root_velocity_to_sim(torch.cat([tank_vel, tank_ang_vel], dim=-1), env_ids)

