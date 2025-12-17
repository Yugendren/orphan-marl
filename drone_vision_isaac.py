"""
Vision-based drone environment integrated with Isaac Sim
Must be run with ISAACSIM_PYTHON
"""

import numpy as np
import asyncio
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw
import time

class DroneVisionIsaacEnv:
    """
    Vision-based environment that runs inside Isaac Sim
    """
    
    def __init__(self, world, config):
        self.world = world
        self.config = config
        
        # Paths
        self.drone_path = config.get('drone_path', '/World/quadrotor')
        self.camera_path = config.get('camera_path', '/World/quadrotor/body/camera')
        self.target_path = config.get('target_path', '/World/SM_Tank_T72b')
        
        # Camera setup
        self.img_height = config.get('img_height', 84)
        self.img_width = config.get('img_width', 84)
        
        # Environment parameters
        self.max_steps = config.get('max_steps', 500)
        self.success_threshold = config.get('success_threshold', 2.0)
        self.max_velocity = config.get('max_velocity', 3.0)
        
        # Detect positions
        self.spawn_pos = self._detect_spawn_position()
        self.target_pos = self._detect_target_position()
        
        print(f"[ENV] DroneVisionIsaacEnv initialized")
        print(f"      Spawn: {self.spawn_pos}")
        print(f"      Target: {self.target_pos}")
        
        # Setup camera
        self._setup_camera()
        
        # State
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.steps = 0
        self.prev_distance = None
        
        # MAVSDK
        self.drone = None
        self.connected = False
        self.loop = None
        
        # Observation/action spaces (for RL Games)
        self.observation_space_shape = (self.img_height, self.img_width, 3)
        self.action_space_shape = (3,)
    
    def _detect_spawn_position(self):
        """Get drone spawn position"""
        from omni.isaac.core.utils.prims import get_prim_at_path
        from pxr import UsdGeom
        
        try:
            drone_prim = get_prim_at_path(self.drone_path)
            if drone_prim and drone_prim.IsValid():
                xform = UsdGeom.Xformable(drone_prim)
                transform = xform.ComputeLocalToWorldTransform(0)
                translation = transform.ExtractTranslation()
                return np.array([float(translation[0]), float(translation[1]), -3.0])
        except:
            pass
        return np.array([0.0, 0.0, -3.0])
    
    def _detect_target_position(self):
        """Get target position"""
        from omni.isaac.core.utils.prims import get_prim_at_path
        from pxr import UsdGeom
        
        try:
            target_prim = get_prim_at_path(self.target_path)
            if target_prim and target_prim.IsValid():
                xform = UsdGeom.Xformable(target_prim)
                transform = xform.ComputeLocalToWorldTransform(0)
                translation = transform.ExtractTranslation()
                return np.array([float(translation[0]), float(translation[1]), -1.0])
        except:
            pass
        return np.array([10.0, 10.0, -1.0])
    
    def _setup_camera(self):
        """Setup camera for rendering"""
        import omni.replicator.core as rep
        
        try:
            # Create render product
            self.render_product = rep.create.render_product(
                self.camera_path,
                (self.img_width, self.img_height)
            )
            
            # Get RGB annotator
            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([self.render_product])
            
            print(f"[ENV] ✓ Camera ready: {self.camera_path}")
        except Exception as e:
            print(f"[ENV] ✗ Camera setup failed: {e}")
            self.render_product = None
            self.rgb_annotator = None
    
    def _get_camera_frame(self):
        """Get current camera frame"""
        if self.rgb_annotator is None:
            # Return blank frame
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        
        try:
            rgb_data = self.rgb_annotator.get_data()
            if rgb_data is not None:
                rgb = np.array(rgb_data)
                # Convert to uint8
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
                return rgb
        except Exception as e:
            print(f"[ENV] Camera capture error: {e}")
        
        return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
    
    def _init_loop(self):
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    async def _connect(self):
        if self.connected:
            return
        
        print("[ENV] Connecting to PX4...")
        self.drone = System()
        await self.drone.connect(system_address="udp://:14540")
        
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[ENV] ✓ Connected to PX4")
                self.connected = True
                break
        
        await self._setup_offboard()
    
    async def _setup_offboard(self):
        async for is_armed in self.drone.telemetry.armed():
            if not is_armed:
                await self.drone.action.arm()
            break
        
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(self.spawn_pos[0], self.spawn_pos[1], self.spawn_pos[2], 0.0))
        
        try:
            await self.drone.offboard.start()
            print("[ENV] ✓ Offboard mode active")
        except:
            pass
        
        await asyncio.sleep(1.0)
    
    async def _get_telemetry(self):
        async for pos_vel in self.drone.telemetry.position_velocity_ned():
            self.drone_pos = np.array([
                pos_vel.position.north_m,
                pos_vel.position.east_m,
                pos_vel.position.down_m
            ])
            self.drone_vel = np.array([
                pos_vel.velocity.north_m_s,
                pos_vel.velocity.east_m_s,
                pos_vel.velocity.down_m_s
            ])
            break
    
    def _compute_reward(self):
        distance = np.linalg.norm(self.target_pos - self.drone_pos)
        
        reward = -distance
        
        if distance < self.success_threshold:
            reward += 100.0
        
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            reward += 20.0 * progress
        
        self.prev_distance = distance
        return float(reward)
    
    def _check_done(self):
        distance = np.linalg.norm(self.target_pos - self.drone_pos)
        
        if distance < self.success_threshold:
            return True, {"success": True}
        
        if self.steps >= self.max_steps:
            return True, {"success": False}
        
        if self.drone_pos[2] > -0.5 or self.drone_pos[2] < -50.0:
            return True, {"success": False}
        
        return False, {}
    
    def reset(self):
        self._init_loop()
        obs = self.loop.run_until_complete(self._async_reset())
        return obs
    
    async def _async_reset(self):
        if not self.connected:
            await self._connect()
        
        # Reset to spawn
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(self.spawn_pos[0], self.spawn_pos[1], self.spawn_pos[2], 0.0))
        
        await asyncio.sleep(2.0)
        
        self.steps = 0
        self.prev_distance = None
        
        # Step simulation
        self.world.step(render=True)
        
        # Get observation
        await self._get_telemetry()
        image = self._get_camera_frame()
        
        distance = np.linalg.norm(self.target_pos - self.drone_pos)
        print(f"[ENV] Reset complete. Distance: {distance:.2f}m")
        
        return image
    
    def step(self, action):
        self._init_loop()
        return self.loop.run_until_complete(self._async_step(action))
    
    async def _async_step(self, action):
        velocity = np.clip(action, -1.0, 1.0) * self.max_velocity
        
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0))
        
        # Step simulation
        self.world.step(render=True)
        
        await asyncio.sleep(0.05)
        
        # Get new state
        await self._get_telemetry()
        image = self._get_camera_frame()
        
        reward = self._compute_reward()
        done, info = self._check_done()
        
        self.steps += 1
        
        if self.steps % 50 == 0:
            distance = np.linalg.norm(self.target_pos - self.drone_pos)
            print(f"[ENV] Step {self.steps}: dist={distance:.2f}m, reward={reward:.2f}")
        
        if done:
            print(f"[ENV] Episode done - success: {info['success']}")
        
        return image, reward, done, info
    
    def close(self):
        if self.connected:
            try:
                self.loop.run_until_complete(self.drone.offboard.stop())
            except:
                pass