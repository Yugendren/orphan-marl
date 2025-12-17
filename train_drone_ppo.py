#!/usr/bin/env python3
"""
| File: train_drone_ppo.py
| Description: Reinforcement Learning training script for a quadrotor drone using Proximal Policy Optimization (PPO).
|
| RL Library: Stable Baselines3 (SB3)
| Algorithm: PPO (Proximal Policy Optimization)
|
| Features:
| - Simulation Environment: NVIDIA Isaac Sim with Pegasus Simulator extension.
| - Task: Navigate the drone to a fixed target position.
| - Observation Spaces:
|   1. State-based: [position, linear_velocity, relative_target_position]
|   2. Vision-based: Monocular camera RGB image (160x120)
| - Action Space: Continuous control [vx, vy, vz, yaw_rate] using a lower-level velocity controller.
|
| Usage:
| - Train (State): python train_drone_ppo.py --mode train
| - Train (Vision): python train_drone_ppo.py --mode train --vision
| - Test: python train_drone_ppo.py --mode test --model-path <path_to_model>
|
| Dependencies:
| - isaacsim
| - pegasus_simulator
| - stable_baselines3
| - gymnasium
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
import sys
import os

# Isaac Sim imports
from isaacsim import SimulationApp

# Launch Isaac Sim
# Check for GUI mode
headless = True
if "--gui" in sys.argv:
    headless = False

# Launch Isaac Sim
simulation_app = SimulationApp({
    "headless": headless,
    "width": 1280,
    "height": 720,
})

# Omniverse imports
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension

# Enable Pegasus extension
enable_extension("pegasus.simulator")

# Pegasus API imports
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.backend import Backend

# Stable Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor


class SimpleRLController(Backend):
    """
    A simple RL controller that applies forces directly to the drone.
    Much faster than PX4 for RL training.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = {}
        super().__init__(config)
        self.current_action = np.zeros(4)  # [vx, vy, vz, yaw_rate]
        self._vehicle_ref = None  # Will be set by the vehicle (use _vehicle_ref to avoid property conflict)
        self._debug_counter = 0
        
        # Simple PD gains for velocity control
        self.hover_thrust = 9.81 * 1.5  # Approximate mass * gravity
        self.kp_xy = 2.0
        self.kp_z = 3.0
        
    def set_vehicle(self, vehicle):
        """Store reference to the vehicle for force application"""
        self._vehicle_ref = vehicle
        
    def update_sensor(self, sensor_type: str, data):
        """Receive sensor data"""
        pass
    
    def update_graphical_sensor(self, sensor_type: str, data):
        """Receive graphical sensor data"""
        pass
    
    def update_state(self, state):
        """Receive vehicle state updates"""
        self._state = state
    
    def input_reference(self):
        """Return control commands - in this case we'll use update() to apply forces"""
        return self.current_action
    
    def update(self, dt: float):
        """Apply forces to drone based on velocity commands"""
        if self._vehicle_ref is None or not hasattr(self._vehicle_ref, 'apply_force'):
            return
            
        # Get current velocity
        vel = self._vehicle_ref.get_linear_velocity()
        if vel is None:
            return
        current_vel = np.array(vel)
        
        # Desired velocity from action
        desired_vel = self.current_action[:3]
        
        # Velocity error
        vel_error = desired_vel - current_vel
        
        # Compute force (P controller)
        force_xy = vel_error[:2] * self.kp_xy
        force_z = vel_error[2] * self.kp_z + self.hover_thrust
        
        # Apply force
        force = np.array([force_xy[0], force_xy[1], force_z])
        self._vehicle_ref.apply_force(force)
    
    def start(self):
        """Start the controller"""
        pass
    
    def stop(self):
        """Stop the controller"""
        pass
    
    def reset(self):
        """Reset the controller"""
        self.current_action = np.zeros(4)
    
    def set_action(self, action):
        """Set the action from RL policy"""
        self.current_action = action


class DroneEnv(gym.Env):
    """
    Gymnasium environment for drone navigation.
    Supports both state-based and vision-based observations.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        use_vision: bool = False,
        max_steps: int = 500,
        target_position: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        render_sim: bool = False
    ):
        super().__init__()
        
        self.use_vision = use_vision
        self.max_steps = max_steps
        self.render_sim = render_sim
        self.target_position = target_position if target_position is not None else np.array([5.0, 0.0, 2.0])
        self.current_step = 0
        
        # Define observation and action spaces
        if use_vision:
            # Vision-based: 160x120 RGB image
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(120, 160, 3), dtype=np.uint8
            )
        else:
            # State-based: [drone_pos(3), drone_vel(3), relative_target(3)]
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
            )
        
        # Action space: [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0, -1.0, -1.0]),
            high=np.array([2.0, 2.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Initialize simulation
        self._setup_simulation()
        
    def _setup_simulation(self):
        """Initialize Isaac Sim world and spawn entities"""
        print("Setting up simulation environment...")
        
        # Create world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # Create PegasusInterface
        self.pg = PegasusInterface()
        self.pg._world = self.world
        
        # Load environment
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # Spawn target (red cube)
        from pxr import UsdGeom, Gf
        stage = self.world.stage
        cube_path = "/World/target_cube"
        cube_prim = UsdGeom.Cube.Define(stage, cube_path)
        cube_prim.GetSizeAttr().Set(0.5)
        cube_prim.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])  # Red
        
        # Set cube position using USD directly
        # Use Xformable API to add translate op
        xform_api = UsdGeom.Xformable(cube_prim)
        xform_api.AddTranslateOp().Set(Gf.Vec3d(
            float(self.target_position[0]),
            float(self.target_position[1]),
            float(self.target_position[2])
        ))
        
        # Create controller
        self.controller = SimpleRLController()
        
        # Spawn drone
        config = MultirotorConfig()
        
        # Add camera if using vision
        if self.use_vision:
            self.camera = MonocularCamera(
                "front_camera",
                config={
                    "resolution": (160, 120),
                    "position": [0.1, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                    "update_rate": 30.0
                }
            )
            config.graphical_sensors = [self.camera]
            
        config.backends = [self.controller]
        
        self.drone = Multirotor(
            stage_prefix="/World/quadrotor",
            usd_file=ROBOTS['Iris'],
            init_pos=[0.0, 0.0, 1.5],
            init_orientation=[0, 0, 0, 1],
            config=config,
        )
        
        # Set vehicle reference in controller
        self.controller.set_vehicle(self.drone)
        
        # Reset world
        self.world.reset()
        
        print(f"Environment ready! Target at {self.target_position}")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.use_vision:
            # Get camera image
            camera_data = self.camera.get_current_frame()
            if camera_data is None or "rgba" not in camera_data:
                return np.zeros((120, 160, 3), dtype=np.uint8)
            
            # Convert RGBA to RGB
            rgba = camera_data["rgba"]
            rgb = rgba[:, :, :3]
            return rgb.astype(np.uint8)
        else:
            # Get state-based observation
            drone_pos = np.array(self.drone.get_world_pose()[0])
            drone_vel = np.array(self.drone.get_linear_velocity())
            relative_target = self.target_position - drone_pos
            
            obs = np.concatenate([drone_pos, drone_vel, relative_target])
            return obs.astype(np.float32)

    def _compute_reward(self) -> Tuple[float, bool]:
        """Compute reward and check if episode is done"""
        drone_pos = np.array(self.drone.get_world_pose()[0])
        distance = np.linalg.norm(drone_pos - self.target_position)
        
        # Distance-based reward
        reward = -distance
        done = False
        
        # Success bonus
        if distance < 0.5:
            reward += 100
            done = True
            print(f"    SUCCESS! Reached target in {self.current_step} steps!")
        
        # Crash penalty
        if drone_pos[2] < 0.1:
            reward -= 50
            done = True
        
        # Too far penalty
        if distance > 15.0:
            reward -= 50
            done = True
        
        # Max steps
        if self.current_step >= self.max_steps:
            done = True
        
        return reward, done
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        
        # Randomize target position
        # Range: x=[-4, 4], y=[-4, 4], z=[1, 3]
        x = np.random.uniform(-4.0, 4.0)
        y = np.random.uniform(-4.0, 4.0)
        z = np.random.uniform(1.0, 3.0)
        self.target_position = np.array([x, y, z])
        
        # Update visual target
        try:
            from pxr import UsdGeom, Gf
            stage = self.world.stage
            cube_prim = stage.GetPrimAtPath("/World/target_cube")
            if cube_prim.IsValid():
                xform_api = UsdGeom.Xformable(cube_prim)
                # Clear existing ops or just update the first translate op if we can find it
                # Taking simpler approach: Resetting the translate op
                # Note: This assumes ADD was used and we can just set it cleanly. 
                # Ideally we iterate ops, but standard way in Isaac examples often just sets it if Xformable.
                
                # More robust: use omni.isaac.core.utils.prims.set_prim_attribute_value ? 
                # Or just use UsdGeom methods.
                
                # Let's clean existing ops to be safe or just use the same API as setup
                xform_api.ClearXformOpOrder()
                xform_api.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
                
                print(f"New Target Position: [{x:.2f}, {y:.2f}, {z:.2f}]")
        except Exception as e:
            print(f"Warning: Could not update target visual: {e}")
        
        # Reset world
        self.world.reset()
        
        # Reset controller
        self.controller.reset()
        
        # Try to reset drone position (may not work due to physics override)
        self.drone.set_world_pose(position=np.array([0.0, 0.0, 1.5]))
        self.drone.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.drone.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        
        # Step physics a few times to stabilize
        for _ in range(10):
            self.world.step(render=False)
        
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment"""
        # Set action in controller
        self.controller.set_action(action)

        should_render = self.render_sim or self.use_vision
        
        # Step physics
        self.world.step(render=should_render)
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward and check if done
        reward, done = self._compute_reward()
        
        self.current_step += 1
        if self.current_step < 20 or self.current_step % 100 == 0:
            print(f"Step: {self.current_step} | Vision: {self.use_vision} | Reward: {reward:.3f} | Done: {done} | Z: {self.drone.get_world_pose()[0][2]:.2f}")
        
        # Info dict
        info = {
            "distance": np.linalg.norm(
                np.array(self.drone.get_world_pose()[0]) - self.target_position
            )
        }
        
        # Gymnasium uses 5-tuple return: (obs, reward, terminated, truncated, info)
        terminated = done and self.current_step < self.max_steps
        truncated = done and self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'world'):
            self.world.stop()
        simulation_app.close()


def make_env(use_vision=False, max_steps=500):
    """Factory function to create environment"""
    def _init():
        env = DroneEnv(use_vision=use_vision, max_steps=max_steps)
        return Monitor(env)
    return _init


def train_ppo(
    use_vision: bool = False,
    total_timesteps: int = 1_000_000,
    save_dir: str = "./drone_ppo_models",
    eval_freq: int = 10_000,
    gui: bool = False
):
    """
    Train PPO agent on drone navigation task
    
    Args:
        use_vision: If True, use vision-based observations. If False, use state-based.
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        eval_freq: Evaluation frequency
        gui: Whether to render the simulation
    """
    
    print("=" * 60)
    print("Drone RL Training with PPO")
    print(f"Observation: {'Vision (160x120 RGB)' if use_vision else 'State (9D)'}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    
    # Create environment
    print("\nCreating environment...")
    env = DroneEnv(use_vision=use_vision, max_steps=500, render_sim=gui)
    
    # Wrap in Monitor for logging
    env = Monitor(env)
    
    # Choose policy based on observation type
    policy = "CnnPolicy" if use_vision else "MlpPolicy"
    
    print(f"\nInitializing PPO with {policy}...")
    
    # Create PPO model
    model = PPO(
        policy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{save_dir}/logs"
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_dir,
        name_prefix="drone_ppo"
    )
    
    print("\nStarting training...\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        # Save final model
        final_path = f"{save_dir}/drone_ppo_final"
        model.save(final_path)
        print(f"Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save(f"{save_dir}/drone_ppo_interrupted")
        print(f"Model saved to: {save_dir}/drone_ppo_interrupted")
    
    finally:
        env.close()


def test_policy(model_path: str, use_vision: bool = False, n_episodes: int = 10):
    """Test a trained policy"""
    print(f"\nTesting policy from {model_path}")
    
    # Create environment
    env = DroneEnv(use_vision=use_vision, max_steps=500)
    
    # Load model
    model = PPO.load(model_path)
    
    # Run episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}, Distance={info['distance']:.2f}m")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train drone with PPO")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"],
                        help="Mode: train or test")
    parser.add_argument("--vision", action="store_true",
                        help="Use vision-based observations (default: state-based)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps (default: 100k)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model for testing")
    parser.add_argument("--gui", action="store_true",
                        help="Run with GUI (disable headless mode)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_ppo(
            use_vision=args.vision,
            total_timesteps=args.timesteps,
            save_dir="./drone_ppo_models",
            gui=args.gui
        )
    else:
        if args.model_path is None:
            print("Error: --model-path required for testing")
            sys.exit(1)
        test_policy(args.model_path, use_vision=args.vision, n_episodes=10)