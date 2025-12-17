"""
| File: train_drone_to_target.py
| Description: RL training script to train a drone to fly towards a red target cube using vision-based RL.
|
| Features:
| - Spawns a multirotor drone (Iris) and a target cube in Isaac Sim.
| - Implements a simple P-controller (acting as the "RL agent" for now) to navigate to the target.
| - Calculates reward based on distance to target.
| - Resets the drone position and target for each episode.
|
| Validation:
| - Validates that the Pegasus Simulator/Isaac Sim environment loads correctly.
| - Validates that the drone can takeoff and apply forces using the `apply_force` API.
| - Validates that sensor data (position/velocity) can be retrieved.
| - Validates the reset logic for continuous training episodes.
"""

import numpy as np
import carb
from isaacsim import SimulationApp

# Start Isaac Sim in HEADLESS mode for training (faster)
simulation_app = SimulationApp({"headless": True})

# -----------------------------------
# Imports after SimulationApp
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.objects import DynamicCuboid
from scipy.spatial.transform import Rotation

# Pegasus API imports
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.backends.backend import Backend


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
        current_vel = np.array(self._vehicle_ref.get_linear_velocity())
        
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
        
        # Debug print
        self._debug_counter += 1
        if self._debug_counter % 300 == 0:
            print(f"      [Controller] Applying force: {force}, vel_error: {vel_error}")
    
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


class DroneRLTrainer:
    """
    Main RL training class for drone navigation.
    """
    
    def __init__(self, num_episodes=1000, max_steps_per_episode=500):
        """Initialize the RL trainer"""
        
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Acquire timeline
        self.timeline = omni.timeline.get_timeline_interface()
        
        # Start Pegasus Interface
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # Load environment
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        
        # Spawn RED TARGET CUBE
        self.target_position = np.array([5.0, 0.0, 2.0])
        self.target = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/target_cube",
                name="target",
                position=self.target_position,
                scale=np.array([0.5, 0.5, 0.5]),
                size=1.0,
                color=np.array([255, 0, 0]),  # Red
            )
        )
        
        # Create drone with camera and RL controller
        self.controller = SimpleRLController()
        
        config_multirotor = MultirotorConfig()
        config_multirotor.backends = [self.controller]
        
        # Add camera sensor (84x84 for RL, common resolution)
        config_multirotor.graphical_sensors = [
            MonocularCamera(
                "camera", 
                config={
                    "update_rate": 30.0,  # 30 FPS
                    "resolution": [84, 84],
                }
            )
        ]
        
        # Spawn drone
        self.drone_start_pos = np.array([0.0, 0.0, 1.5])
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            self.drone_start_pos.tolist(),
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )
        
        # Pass vehicle reference to controller for force application
        self.controller.set_vehicle(self.drone)
        
        # Initialize the world
        self.world.reset()
        self.timeline.play()
        
        print("=" * 50)
        print("Drone RL Trainer Initialized!")
        print(f"Training for {num_episodes} episodes")
        print(f"Target: Red cube at {self.target_position}")
        print("=" * 50)
    
    def get_observation(self):
        """
        Get current observation for RL policy.
        For now, returns: [drone_pos, drone_vel, relative_target_position]
        Later, can use camera images for vision-based RL.
        """
        # Get drone state
        drone_pos = np.array(self.drone.state.position)
        drone_vel = np.array(self.drone.state.linear_velocity)
        
        # Relative position to target
        relative_target = self.target_position - drone_pos
        
        # Concatenate into observation vector
        obs = np.concatenate([
            drone_pos,           # [x, y, z]
            drone_vel,           # [vx, vy, vz]
            relative_target,     # [dx, dy, dz]
        ])
        
        return obs
    
    def get_camera_image(self):
        """
        Get camera image from drone.
        Returns RGB image as numpy array.
        """
        # Access camera data through Pegasus API
        # This will be available after world.step()
        camera = self.drone.sensors[0]  # First sensor is the camera
        
        # For vision-based RL, you'd process this image
        # For now, we'll use state-based observations
        return None
    
    def calculate_reward(self, drone_pos):
        """
        Calculate reward based on distance to target.
        """
        distance_to_target = np.linalg.norm(self.target_position - drone_pos)
        
        # Reward shaping: negative distance (closer = better)
        reward = -distance_to_target
        
        # Bonus for reaching target
        if distance_to_target < 0.5:
            reward += 100.0
            done = True
            print(f"    TARGET REACHED! Distance: {distance_to_target:.3f}m")
        else:
            done = False
        
        # Penalty for going too far or too low
        if drone_pos[2] < 0.1:
            reward -= 50.0
            done = True
            print(f"    CRASHED! Height: {drone_pos[2]:.3f}m, Distance: {distance_to_target:.2f}m, Reward: {reward:.2f}")
        elif distance_to_target > 15.0:
            reward -= 50.0
            done = True
            print(f"    TOO FAR! Distance: {distance_to_target:.2f}m, Reward: {reward:.2f}")
        
        return reward, done
    
    def reset_episode(self):
        """Reset environment for new episode"""
        # Generate new random position
        random_offset = np.random.uniform(-1.0, 1.0, size=3)
        random_offset[2] = np.random.uniform(0.5, 2.0)  # Keep z positive
        
        new_pos = self.drone_start_pos + random_offset
        
        print(f"  Generated offset: [{random_offset[0]:.2f}, {random_offset[1]:.2f}, {random_offset[2]:.2f}]")
        print(f"  Target position: [{new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f}]")
        
        # Use Pegasus's built-in methods (the proper way!)
        # Set position using set_world_pose
        self.drone.set_world_pose(
            position=new_pos,
            orientation=np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (w,x,y,z)
        )
        
        # Reset velocities using set_linear_velocity and set_angular_velocity
        self.drone.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
        self.drone.set_angular_velocity(np.array([0.0, 0.0, 0.0]))
        
        # Call post_reset if it exists (might do additional cleanup)
        if hasattr(self.drone, 'post_reset'):
            self.drone.post_reset()
        
        # Reset controller
        self.controller.reset()
        
        # Step physics to apply changes
        for _ in range(10):
            self.world.step(render=False)
        
        # Verify the reset worked
        actual_pos, _ = self.drone.get_world_pose()
        distance = np.linalg.norm(self.target_position - actual_pos)
        print(f"  Actual: Drone at [{actual_pos[0]:.2f}, {actual_pos[1]:.2f}, {actual_pos[2]:.2f}], distance: {distance:.2f}m")
    
    def random_policy(self, obs):
        """
        Random policy for testing.
        Replace this with your trained RL policy.
        """
        # Random velocity commands: [vx, vy, vz, yaw_rate]
        action = np.random.uniform(-1.0, 1.0, size=4)
        return action
    
    def simple_heuristic_policy(self, obs):
        """
        Simple heuristic policy: fly towards target.
        Useful for debugging and baseline.
        """
        # Extract relative target position from observation
        relative_target = obs[6:9]  # Last 3 elements
        
        # Simple proportional control
        action = np.zeros(4)
        action[0:3] = np.clip(relative_target * 0.5, -2.0, 2.0)  # velocity command
        action[3] = 0.0  # no yaw
        
        return action
    
    def train(self):
        """
        Main training loop.
        """
        
        for episode in range(self.num_episodes):
            # Reset environment
            self.reset_episode()
            
            episode_reward = 0.0
            episode_steps = 0
            
            for step in range(self.max_steps_per_episode):
                # Get observation
                obs = self.get_observation()
                
                # Get action from policy
                # TODO: Replace with your RL algorithm (PPO, SAC, etc.)
                action = self.simple_heuristic_policy(obs)  # Use random_policy() for random
                
                # Debug: Print action and state every 30 steps
                if step % 30 == 0 and episode == 0:
                    drone_pos = np.array(self.drone.state.position)
                    vel = np.array(self.drone.state.linear_velocity)
                    print(f"    Step {step}: Action={action}, Pos=[{drone_pos[0]:.2f},{drone_pos[1]:.2f},{drone_pos[2]:.2f}], Vel=[{vel[0]:.2f},{vel[1]:.2f},{vel[2]:.2f}]")
                
                # Apply action
                self.controller.set_action(action)
                
                # Step simulation
                self.world.step(render=False)  # render=False for speed
                
                # Get new state and calculate reward
                new_obs = self.get_observation()
                drone_pos = np.array(self.drone.state.position)
                reward, done = self.calculate_reward(drone_pos)
                
                episode_reward += reward
                episode_steps += 1
                
                # Store transition for RL training
                # TODO: Add to replay buffer or trajectory buffer
                # Example: replay_buffer.add(obs, action, reward, new_obs, done)
                
                if done:
                    break
            
            # Log episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {episode}/{self.num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.1f} steps")
            
            # TODO: Update policy with RL algorithm
            # Example: agent.update()
        
        print("\n" + "=" * 50)
        print("Training Complete!")
        print(f"Final Avg Reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print("=" * 50)
    
    def close(self):
        """Cleanup and close simulation"""
        self.timeline.stop()
        simulation_app.close()


def main():
    """Main entry point"""
    
    # Create trainer
    trainer = DroneRLTrainer(
        num_episodes=100,  # Start with 100 episodes
        max_steps_per_episode=500
    )
    
    try:
        # Run training
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Cleanup
        trainer.close()


if __name__ == "__main__":
    main()