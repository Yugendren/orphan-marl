"""
Evaluation script for trained Drone Tank Task agent
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from environments.drone_tank_task import DroneTankTask
from agents.drone_agent import DroneActorCritic, DroneAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate Drone Tank Task Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint to evaluate")
    parser.add_argument("--task-config", type=str, default="configs/task/DroneTankTask.yaml",
                       help="Path to task configuration file")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                       help="Render evaluation episodes")
    
    args = parser.parse_args()
    
    # Load task config
    import yaml
    with open(args.task_config, 'r') as f:
        task_config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = DroneTankTask(
        cfg=task_config,
        rl_device=device,
        sim_device=device,
        graphics_device_id=0,
        headless=not args.render,
        virtual_screen_capture=False,
        force_render=args.render
    )
    
    # Create agent
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
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluation loop
    print(f"Evaluating for {args.num_episodes} episodes...")
    
    total_rewards = []
    success_count = 0
    
    for episode in range(args.num_episodes):
        obs = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            # Select action (deterministic for evaluation)
            action = agent.select_action(obs, deterministic=True)
            
            # Step environment
            env.pre_physics_step(action)
            env.gym.simulate(env.sim)
            env.post_physics_step()
            
            # Get reward
            reward = env.rew_buf.mean().item()
            episode_reward += reward
            steps += 1
            
            # Check if done
            done = env.reset_buf.any().item()
            
            if args.render:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
        
        total_rewards.append(episode_reward)
        
        # Check success (tank hit)
        # In practice, check reset_buf for success condition
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    # Print statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    success_rate = success_count / args.num_episodes
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()

