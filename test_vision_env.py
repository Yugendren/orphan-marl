#!/usr/bin/env python3
"""
Test vision-based drone RL environment
Shows camera feed and tests the environment
"""

import sys
sys.path.append('/home/ubuntu/drone_rl')

from envs.drone_vision_env import DroneVisionEnv
import time
import cv2
import numpy as np

def test():
    print("="*60)
    print("Testing Vision-Based Drone RL Environment")
    print("="*60)
    print("\nSetup checklist:")
    print("1. Open Isaac Sim")
    print("2. Load your scene with drone and target")
    print("3. Make sure camera is attached to drone")
    print("4. Press Play")
    print("\nPress ENTER to continue...")
    input()
    
    # Find objects first
    print("\n[INFO] Use find_objects.py to discover paths if needed")
    print("Common paths:")
    print("  Drone: /World/quadcopter")
    print("  Camera: /World/quadcopter/body/camera")
    print("  Target: /World/[your_tank_name]\n")
    
    # Get paths from user
    drone_path = input("Enter drone path [/World/quadcopter]: ").strip() or "/World/quadcopter"
    camera_path = input("Enter camera path [/World/quadcopter/body/camera]: ").strip() or "/World/quadcopter/body/camera"
    target_path = input("Enter target path [/World/target]: ").strip() or "/World/target"
    
    print("\n[INFO] Creating environment...")
    
    # Create vision-based environment
    env = DroneVisionEnv({
        'drone_path': drone_path,
        'camera_path': camera_path,
        'target_path': target_path,
        'img_height': 84,
        'img_width': 84,
        'use_grayscale': False,  # Use color images
        'use_state': True,       # Include position/velocity in observation
        'max_steps': 200
    })
    
    # Reset and get initial observation
    print("\n[INFO] Resetting environment...")
    obs, _ = env.reset()
    
    if isinstance(obs, dict):
        image = obs['image']
        state = obs['state']
        print(f"\n✓ Environment ready!")
        print(f"   Image shape: {image.shape}")
        print(f"   State shape: {state.shape}")
        print(f"   Drone position: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]")
    else:
        image = obs
        print(f"\n✓ Environment ready!")
        print(f"   Image shape: {image.shape}")
    
    # Create window for camera feed
    cv2.namedWindow('Drone Camera Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Drone Camera Feed', 420, 420)  # 5x larger for visibility
    
    print("\n[INFO] Running random actions...")
    print("Watch the camera feed window!")
    print("Press 'q' in camera window to stop early\n")
    
    # Run episode with visualization
    for i in range(100):
        # Random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        
        # Extract image from observation
        if isinstance(obs, dict):
            image = obs['image']
            state = obs['state']
        else:
            image = obs
        
        # Display camera feed
        display_image = cv2.resize(image, (420, 420))  # Upscale for viewing
        
        # Add text overlay
        cv2.putText(display_image, f"Step: {i}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_image, f"Reward: {reward:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert RGB to BGR for OpenCV
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Drone Camera Feed', display_image)
        
        # Print status
        if i % 10 == 0:
            if isinstance(obs, dict):
                distance = np.linalg.norm(env.target_pos - state[:3])
                print(f"Step {i:3d}: reward={reward:6.2f}, distance={distance:6.2f}m, "
                      f"pos=[{state[0]:5.1f}, {state[1]:5.1f}, {state[2]:5.1f}]")
            else:
                print(f"Step {i:3d}: reward={reward:6.2f}")
        
        # Check for done
        if done:
            print(f"\n✓ Episode finished: {info['reason']}")
            if info.get('success', False):
                print("🎉 SUCCESS - Target reached!")
            break
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Stopped by user")
            break
        
        time.sleep(0.05)
    
    # Keep final image visible
    print("\n[INFO] Episode complete. Press any key in camera window to close...")
    cv2.waitKey(0)
    
    # Cleanup
    cv2.destroyAllWindows()
    env.close()
    print("\n✓ Test complete!")
    print("\nNext steps:")
    print("1. If camera feed looked good → proceed to training")
    print("2. If no camera feed → check camera path")
    print("3. If wrong view → adjust camera position/rotation in Isaac Sim")

if __name__ == '__main__':
    test()