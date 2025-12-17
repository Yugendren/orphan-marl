#!/usr/bin/env python3
"""
Minimal vision test - just camera, no RL
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage
import omni.replicator.core as rep
import numpy as np
from PIL import Image

print("="*60)
print("Minimal Vision Test")
print("="*60)

# Load scene
scene_path = "/home/ubuntu/drone_rl/scenes/drone_env_2.usd"
print(f"\n[INFO] Loading: {scene_path}")

try:
    open_stage(scene_path)
    print("[INFO] ✓ Scene loaded")
except Exception as e:
    print(f"[ERROR] Could not load scene: {e}")
    print("\nMake sure the scene file exists!")
    simulation_app.close()
    exit(1)

# Create world
world = World()
world.reset()

# Setup camera
camera_path = "/World/quadrotor/body/camera"
print(f"\n[INFO] Setting up camera: {camera_path}")

try:
    render_product = rep.create.render_product(camera_path, (84, 84))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    print("[INFO] ✓ Camera ready")
except Exception as e:
    print(f"[ERROR] Camera setup failed: {e}")
    simulation_app.close()
    exit(1)

# Capture frames
print("\n[INFO] Capturing 5 frames...")

for i in range(5):
    # Step simulation
    world.step(render=True)
    
    try:
        # Get frame
        rgb_data = rgb_annotator.get_data()
        
        if rgb_data is not None:
            rgb = np.array(rgb_data)
            
            # Convert to uint8
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)
            
            # Save
            img = Image.fromarray(rgb)
            filepath = f"/home/ubuntu/drone_rl/minimal_frame_{i}.png"
            img.save(filepath)
            
            print(f"  ✓ Frame {i+1}/5: {filepath} - shape: {rgb.shape}")
        else:
            print(f"  ✗ Frame {i+1}/5: No data returned")
    
    except Exception as e:
        print(f"  ✗ Frame {i+1}/5: Error - {e}")

print("\n✓ Test complete!")
print("\nCheck frames: ls ~/drone_rl/minimal_frame_*.png")

simulation_app.close()