"""
Training script for Drone Tank Task
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.drone_tank_task import DroneTankTask
from agents.drone_agent import DroneActorCritic, DroneAgent
from rewards.reward_functions import (
    TankHitReward, 
    GroundCrashPenalty, 
    DistanceReward, 
    SurvivalReward,
    CombinedReward
)


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_reward_function(config: dict, device: str = "cuda:0"):
    """Create combined reward function from config."""
    reward_config = config.get("reward", {})
    
    reward_functions = {
        "tank_hit": TankHitReward({
            "reward_value": reward_config.get("tank_hit", 10.0),
            "contact_threshold": reward_config.get("tank_contact_threshold", 0.5),
            "use_distance": reward_config.get("use_distance_reward", False),
        }),
        "ground_crash": GroundCrashPenalty({
            "penalty_value": reward_config.get("ground_crash", -5.0),
            "contact_threshold": reward_config.get("ground_contact_threshold", 0.3),
            "ground_height": reward_config.get("ground_height", 0.0),
            "device": device,
        }),
        "distance": DistanceReward({
            "scale": reward_config.get("distance_scale", 0.1),
            "max_distance": reward_config.get("max_distance", 20.0),
        }),
        "survival": SurvivalReward({
            "reward_value": reward_config.get("survival", 0.01),
            "device": device,
        }),
    }
    
    weights = {
        "tank_hit": reward_config.get("tank_hit_weight", 1.0),
        "ground_crash": reward_config.get("ground_crash_weight", 1.0),
        "distance": reward_config.get("distance_weight", 1.0),
        "survival": reward_config.get("survival_weight", 1.0),
    }
    
    return CombinedReward(reward_functions, weights)


def main():
    parser = argparse.ArgumentParser(description="Train Drone Tank Task")
    parser.add_argument("--task-config", type=str, default="configs/task/DroneTankTask.yaml",
                       help="Path to task configuration file")
    parser.add_argument("--train-config", type=str, default="configs/train/DroneTankPPO.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory for checkpoints and logs")
    
    args = parser.parse_args()
    
    # Load configurations
    task_config = load_config(args.task_config)
    train_config = load_config(args.train_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = DroneTankTask(
        cfg=task_config,
        rl_device=device,
        sim_device=device,
        graphics_device_id=0,
        headless=args.headless,
        virtual_screen_capture=False,
        force_render=False
    )
    
    # Create agent
    print("Creating agent...")
    state_dim = env.num_state_obs
    vision_feature_dim = 256
    action_dim = env.num_actions
    
    model = DroneActorCritic(
        state_dim=state_dim,
        vision_feature_dim=vision_feature_dim,
        action_dim=action_dim,
        use_vision=task_config.get("observations", {}).get("camera", {}).get("enabled", True)
    )
    
    action_scale = torch.tensor(
        task_config.get("actions", {}).get("scale", [1.0] * action_dim),
        device=device
    )
    
    agent = DroneAgent(model, action_scale, device)
    
    # Create reward function
    reward_fn = create_reward_function(task_config, str(device))
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
    
    # Training loop (simplified - in practice, use RL-Games or similar)
    print("Starting training...")
    print("Note: This is a simplified training script.")
    print("For full training, integrate with RL-Games or similar RL framework.")
    
    # Placeholder training loop
    max_iterations = train_config.get("params", {}).get("trainer", {}).get("max_iterations", 10000)
    
    for iteration in range(max_iterations):
        # Get observations
        obs = env.reset() if iteration == 0 else env.obs_buf
        
        # Select actions
        # Note: In practice, you'd need to handle vision observations here
        actions = agent.select_action(obs)
        
        # Step environment
        env.pre_physics_step(actions)
        env.gym.simulate(env.sim)
        env.post_physics_step()
        
        # Get rewards
        rewards = env.rew_buf
        
        # Logging
        if iteration % 100 == 0:
            avg_reward = rewards.mean().item()
            print(f"Iteration {iteration}: Average reward = {avg_reward:.4f}")
        
        # Save checkpoint
        if iteration % 500 == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_{iteration}.pt")
            torch.save({
                "iteration": iteration,
                "model_state_dict": agent.model.state_dict(),
                "config": task_config,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()

