"""
Simplified drone RL environment with automatic position detection
Assumes Isaac Sim scene is already set up and running
"""

import gymnasium as gym
import numpy as np
import asyncio
from gymnasium import spaces
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw

# Isaac Sim imports for getting object positions
try:
    from omni.isaac.core.utils.prims import get_prim_at_path
    from pxr import UsdGeom, Gf
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    print("[WARNING] Isaac Sim Python API not available. Using manual coordinates.")

class SimpleDroneEnv(gym.Env):
    """
    Minimal RL environment - scene is pre-configured in Isaac Sim
    Automatically detects drone and target positions from the scene
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        config = config or {}
        
        # Paths to objects in Isaac Sim scene
        self.drone_path = config.get("drone_path", "/World/quadcopter")
        self.target_path = config.get("target_path", "/World/target")  # Update this!
        
        # Auto-detect positions or use provided defaults
        if ISAAC_AVAILABLE and config.get("auto_detect_positions", True):
            detected_spawn, detected_target = self._detect_positions()
            self.spawn_pos = detected_spawn
            self.target_pos = detected_target
            print(f"[ENV] Auto-detected positions:")
            print(f"      Drone spawn: {self.spawn_pos}")
            print(f"      Target: {self.target_pos}")
        else:
            # Fallback to manual coordinates
            self.target_pos = np.array(config.get("target_pos", [10.0, 10.0, -1.0]))
            self.spawn_pos = np.array(config.get("spawn_pos", [0.0, 0.0, -3.0]))
            print(f"[ENV] Using manual coordinates:")
            print(f"      Spawn: {self.spawn_pos}")
            print(f"      Target: {self.target_pos}")
        
        self.max_steps = config.get("max_steps", 500)
        self.success_threshold = config.get("success_threshold", 2.0)
        self.max_velocity = config.get("max_velocity", 3.0)
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # State
        self.drone_pos = np.zeros(3)
        self.drone_vel = np.zeros(3)
        self.steps = 0
        self.prev_distance = None
        
        # MAVSDK
        self.drone = None
        self.connected = False
        self.loop = None
    
    def _detect_positions(self):
        """
        Automatically detect drone and target positions from Isaac Sim scene
        Returns: (spawn_pos, target_pos) as numpy arrays
        """
        try:
            # Get drone position
            drone_prim = get_prim_at_path(self.drone_path)
            if drone_prim and drone_prim.IsValid():
                xform = UsdGeom.Xformable(drone_prim)
                transform = xform.ComputeLocalToWorldTransform(0)
                translation = transform.ExtractTranslation()
                
                # Isaac Sim coordinates to NED (drone spawn at altitude)
                spawn_x = float(translation[0])
                spawn_y = float(translation[1])
                spawn_z = -3.0  # Always spawn at 3m altitude
                
                spawn_pos = np.array([spawn_x, spawn_y, spawn_z])
            else:
                print(f"[WARNING] Drone not found at {self.drone_path}, using default")
                spawn_pos = np.array([0.0, 0.0, -3.0])
            
            # Get target position
            target_prim = get_prim_at_path(self.target_path)
            if target_prim and target_prim.IsValid():
                xform = UsdGeom.Xformable(target_prim)
                transform = xform.ComputeLocalToWorldTransform(0)
                translation = transform.ExtractTranslation()
                
                # Isaac Sim coordinates to NED
                target_x = float(translation[0])
                target_y = float(translation[1])
                target_z = -1.0  # Target at ~1m altitude
                
                target_pos = np.array([target_x, target_y, target_z])
            else:
                print(f"[WARNING] Target not found at {self.target_path}, using default")
                target_pos = np.array([10.0, 10.0, -1.0])
            
            return spawn_pos, target_pos
            
        except Exception as e:
            print(f"[ERROR] Position detection failed: {e}")
            print("[ENV] Using default positions")
            return np.array([0.0, 0.0, -3.0]), np.array([10.0, 10.0, -1.0])
    
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
                print("[ENV] ✓ Connected!")
                self.connected = True
                break
        
        # Setup offboard
        await self._setup_offboard()
    
    async def _setup_offboard(self):
        # Arm
        async for is_armed in self.drone.telemetry.armed():
            if not is_armed:
                await self.drone.action.arm()
            break
        
        # Start offboard
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(
                self.spawn_pos[0],
                self.spawn_pos[1],
                self.spawn_pos[2],
                0.0
            ))
        
        try:
            await self.drone.offboard.start()
            print("[ENV] ✓ Offboard mode active")
        except:
            pass  # Might already be active
        
        await asyncio.sleep(1.0)
    
    async def _get_telemetry(self):
        """Get drone state"""
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
    
    def _get_obs(self):
        relative = self.target_pos - self.drone_pos
        distance = np.linalg.norm(relative)
        
        return np.concatenate([
            self.drone_pos,
            self.drone_vel,
            relative,
            [distance]
        ]).astype(np.float32)
    
    def _compute_reward(self):
        distance = np.linalg.norm(self.target_pos - self.drone_pos)
        
        # Main reward: negative distance
        reward = -distance
        
        # Success bonus
        if distance < self.success_threshold:
            reward += 100.0
        
        # Progress reward
        if self.prev_distance is not None:
            progress = self.prev_distance - distance
            reward += 20.0 * progress
        
        self.prev_distance = distance
        return float(reward)
    
    def _check_done(self):
        distance = np.linalg.norm(self.target_pos - self.drone_pos)
        
        if distance < self.success_threshold:
            return True, {"success": True, "reason": "reached_target"}
        
        if self.steps >= self.max_steps:
            return True, {"success": False, "reason": "timeout"}
        
        if self.drone_pos[2] > -0.5 or self.drone_pos[2] < -50.0:
            return True, {"success": False, "reason": "crash"}
        
        return False, {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_loop()
        obs = self.loop.run_until_complete(self._async_reset())
        return obs, {}
    
    async def _async_reset(self):
        if not self.connected:
            await self._connect()
        
        # Reset to spawn position
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(
                self.spawn_pos[0],
                self.spawn_pos[1],
                self.spawn_pos[2],
                0.0
            ))
        
        await asyncio.sleep(2.0)
        
        self.steps = 0
        self.prev_distance = None
        
        await self._get_telemetry()
        obs = self._get_obs()
        
        print(f"[ENV] Reset - Distance to target: {obs[-1]:.2f}m")
        return obs
    
    def step(self, action):
        self._init_loop()
        return self.loop.run_until_complete(self._async_step(action))
    
    async def _async_step(self, action):
        velocity = np.clip(action, -1.0, 1.0) * self.max_velocity
        
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(
                float(velocity[0]),
                float(velocity[1]),
                float(velocity[2]),
                0.0
            ))
        
        await asyncio.sleep(0.05)
        
        await self._get_telemetry()
        reward = self._compute_reward()
        done, info = self._check_done()
        
        self.steps += 1
        obs = self._get_obs()
        
        if self.steps % 50 == 0:
            print(f"[ENV] Step {self.steps}: dist={obs[-1]:.2f}m, reward={reward:.2f}")
        
        if done:
            print(f"[ENV] Episode done - {info['reason']}")
        
        return obs, reward, done, False, info
    
    def close(self):
        if self.connected:
            try:
                self.loop.run_until_complete(self.drone.offboard.stop())
            except:
                pass