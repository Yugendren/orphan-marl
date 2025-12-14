"""
PPO Training script for Drone Tank Task using Isaac Lab
Migrated to use Isaac Lab APIs
"""

import os
import sys
import argparse
import torch
import torch.optim as optim
import numpy as np
from collections import deque
from pathlib import Path

# Add project root and Isaac Lab to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add Isaac Lab to path
isaac_lab_path = Path("/home/ubuntu/IsaacLab")
if isaac_lab_path.exists():
    sys.path.insert(0, str(isaac_lab_path))
    sys.path.insert(0, str(isaac_lab_path / "source"))

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Train Drone Tank Task with PPO (Isaac Lab)")
parser.add_argument("--task-config", type=str, default=None,
                   help="Path to task configuration file (optional, uses default if not provided)")
parser.add_argument("--train-config", type=str, default="configs/train/DroneTankPPO.yaml",
                   help="Path to training configuration file")
# Note: --headless is provided by AppLauncher, don't add it here
parser.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint to resume from")
parser.add_argument("--output-dir", type=str, default="outputs",
                   help="Output directory for checkpoints and logs")
parser.add_argument("--num-envs", type=int, default=4096,
                   help="Number of parallel environments")

# Append AppLauncher cli args (includes --headless)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Workaround for inspect.getfile() issue with namespace packages
import inspect
_orig_getfile = inspect.getfile
def _patched_getfile(object):
    try:
        return _orig_getfile(object)
    except TypeError:
        # Handle namespace packages (like isaaclab)
        if hasattr(object, '__path__'):
            # Return a dummy path for namespace packages
            return object.__path__[0] if object.__path__ else '<namespace>'
        raise
inspect.getfile = _patched_getfile

import yaml
from environments.drone_tank_task_isaaclab import DroneTankTask, DroneTankTaskCfg
from agents.drone_agent import DroneActorCritic, DroneAgent


class ExperienceBuffer:
    """Buffer for storing experience tuples."""
    
    def __init__(self, num_envs, state_dim, action_dim, max_size, device):
        self.max_size = max_size
        self.device = device
        self.num_envs = num_envs
        
        # Buffers
        self.states = torch.zeros((max_size, num_envs, state_dim), device=device)
        self.actions = torch.zeros((max_size, num_envs, action_dim), device=device)
        self.rewards = torch.zeros((max_size, num_envs), device=device)
        self.values = torch.zeros((max_size, num_envs), device=device)
        self.log_probs = torch.zeros((max_size, num_envs), device=device)
        self.dones = torch.zeros((max_size, num_envs), device=device, dtype=torch.bool)
        self.images = None  # Will be set if vision is used
        
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, value, log_prob, done, images=None):
        """Add experience to buffer."""
        idx = self.ptr % self.max_size
        
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.values[idx] = value
        self.log_probs[idx] = log_prob
        self.dones[idx] = done
        
        if images is not None:
            if self.images is None:
                # Initialize image buffer
                img_shape = images.shape[1:]  # (num_envs, C, H, W)
                self.images = torch.zeros((self.max_size, *img_shape), device=self.device)
            self.images[idx] = images
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def get_all(self):
        """Get all experiences."""
        if self.size < self.max_size:
            indices = torch.arange(self.size, device=self.device)
        else:
            indices = torch.arange(self.max_size, device=self.device)
        
        states = self.states[indices].view(-1, self.states.shape[-1])
        actions = self.actions[indices].view(-1, self.actions.shape[-1])
        rewards = self.rewards[indices].view(-1)
        values = self.values[indices].view(-1)
        log_probs = self.log_probs[indices].view(-1)
        dones = self.dones[indices].view(-1)
        
        images = None
        if self.images is not None:
            images = self.images[indices].view(-1, *self.images.shape[2:])
        
        return states, actions, rewards, values, log_probs, dones, images
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0
        self.size = 0


def compute_gae(rewards, values, dones, gamma=0.99, tau=0.95):
    """Compute Generalized Advantage Estimation (GAE)."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (~dones[t]).float() - values[t]
        gae = delta + gamma * tau * (~dones[t]).float() * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    
    return advantages, returns


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment configuration
    print("Creating environment configuration...")
    env_cfg = DroneTankTaskCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = str(device)
    
    # Load training config if provided
    train_config = {}
    if args_cli.train_config and os.path.exists(args_cli.train_config):
        train_config = load_config(args_cli.train_config)
    
    # Create environment
    print("Creating environment...")
    env = DroneTankTask(cfg=env_cfg, render_mode=None)
    
    # Create agent
    print("Creating agent...")
    # Get observation space dimension
    obs_space = env.single_observation_space["policy"]
    if hasattr(obs_space, 'shape'):
        state_dim = obs_space.shape[0]
    elif isinstance(obs_space, dict):
        # If dict, get state dimension
        if "state" in obs_space:
            state_dim = obs_space["state"].shape[0]
        else:
            # Flatten all observations
            state_dim = sum(s.shape[0] if hasattr(s, 'shape') else 1 for s in obs_space.values())
    else:
        state_dim = 13  # Default
    
    vision_feature_dim = 256
    action_dim = env.single_action_space.shape[0] if hasattr(env.single_action_space, 'shape') else 4
    
    use_vision = env.use_vision
    
    model = DroneActorCritic(
        state_dim=state_dim,
        vision_feature_dim=vision_feature_dim,
        action_dim=action_dim,
        use_vision=use_vision
    )
    
    action_scale = env_cfg.action_scale.to(device)
    agent = DroneAgent(model, action_scale, device)
    
    # Optimizer
    algo_config = train_config.get("params", {}).get("algo", {}).get("ppo", {})
    learning_rate = algo_config.get("learning_rate", 3.0e-4)
    optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)
    
    # Load checkpoint if provided
    start_iteration = 0
    if args_cli.checkpoint:
        print(f"Loading checkpoint from {args_cli.checkpoint}")
        checkpoint = torch.load(args_cli.checkpoint, map_location=device)
        agent.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint.get("iteration", 0)
    
    # Training parameters
    trainer_config = train_config.get("params", {}).get("trainer", {})
    max_iterations = trainer_config.get("max_iterations", 10000)
    rollout_length = algo_config.get("rollout_length", 2048)
    num_learning_epochs = algo_config.get("num_learning_epochs", 5)
    num_mini_batches = algo_config.get("num_mini_batches", 4)
    clip_param = algo_config.get("clip_param", 0.2)
    value_loss_coef = algo_config.get("value_loss_coef", 0.5)
    entropy_coef = algo_config.get("entropy_coef", 0.01)
    gamma = algo_config.get("gamma", 0.99)
    tau = algo_config.get("tau", 0.95)
    max_grad_norm = algo_config.get("max_grad_norm", 1.0)
    
    # Adjust rollout length if too large
    if rollout_length > env.num_envs * 10:
        rollout_length = env.num_envs * 10
        print(f"Adjusted rollout_length to {rollout_length} based on num_envs")
    
    # Experience buffer
    buffer = ExperienceBuffer(
        num_envs=env.num_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=rollout_length,
        device=device
    )
    
    print("Starting PPO training...")
    print(f"Rollout length: {rollout_length}, Learning epochs: {num_learning_epochs}")
    print(f"Mini batches: {num_mini_batches}, Clip param: {clip_param}")
    print("="*60)
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    if isinstance(obs, dict):
        state_obs = obs.get("state", obs)
        images = obs.get("camera", None)
    else:
        state_obs = obs
        images = None
    
    for iteration in range(start_iteration, max_iterations):
        # Collect rollout
        buffer.clear()
        
        for step in range(rollout_length):
            # Get action from policy
            with torch.no_grad():
                if images is not None:
                    action, log_prob = agent.model.get_action(state_obs, images)
                    _, _, value = agent.model.forward(state_obs, images)
                else:
                    action, log_prob = agent.model.get_action(state_obs)
                    _, _, value = agent.model.forward(state_obs)
                
                # Scale action
                scaled_action = action * action_scale
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(scaled_action)
            
            # Get new observations
            new_obs = obs_dict["policy"]
            if isinstance(new_obs, dict):
                new_state_obs = new_obs.get("state", new_obs)
                new_images = new_obs.get("camera", None)
            else:
                new_state_obs = new_obs
                new_images = None
            
            done = terminated | truncated
            
            # Store experience
            buffer.add(state_obs, scaled_action, reward, value.squeeze(-1), 
                      log_prob.squeeze(-1), done, images)
            
            # Update for next step
            state_obs = new_state_obs
            images = new_images
        
        # Compute advantages and returns
        states, actions, rewards, values, old_log_probs, dones, buffer_images = buffer.get_all()
        
        # Reshape for GAE computation
        T = rollout_length
        N = env.num_envs
        rewards_reshaped = rewards.view(T, N)
        values_reshaped = values.view(T, N)
        dones_reshaped = dones.view(T, N)
        
        # Compute GAE
        advantages, returns = compute_gae(rewards_reshaped, values_reshaped, 
                                          dones_reshaped, gamma, tau)
        
        # Flatten for training
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Learning phase
        batch_size = len(states)
        mini_batch_size = batch_size // num_mini_batches
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(num_learning_epochs):
            # Shuffle data
            indices = torch.randperm(batch_size, device=device)
            
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_images = buffer_images[batch_indices] if buffer_images is not None else None
                
                # Evaluate actions
                if batch_images is not None:
                    log_probs, new_values, entropy = agent.model.evaluate_actions(
                        batch_states, batch_images, batch_actions
                    )
                else:
                    log_probs, new_values, entropy = agent.model.evaluate_actions(
                        batch_states, None, batch_actions
                    )
                
                log_probs = log_probs.squeeze(-1)
                new_values = new_values.squeeze(-1)
                entropy = entropy.squeeze(-1)
                
                # PPO policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = ((new_values - batch_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_grad_norm)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Logging
        avg_reward = rewards.mean().item()
        avg_advantage = advantages.mean().item()
        avg_policy_loss = total_policy_loss / (num_learning_epochs * num_mini_batches)
        avg_value_loss = total_value_loss / (num_learning_epochs * num_mini_batches)
        avg_entropy = total_entropy / (num_learning_epochs * num_mini_batches)
        
        if iteration % 10 == 0:
            print(f"Iteration {iteration}:")
            print(f"  Avg Reward: {avg_reward:.4f}")
            print(f"  Avg Advantage: {avg_advantage:.4f}")
            print(f"  Policy Loss: {avg_policy_loss:.4f}")
            print(f"  Value Loss: {avg_value_loss:.4f}")
            print(f"  Entropy: {avg_entropy:.4f}")
            print("-" * 60)
        
        # Save checkpoint
        if iteration % 500 == 0 and iteration > 0:
            checkpoint_path = os.path.join(args_cli.output_dir, f"checkpoint_{iteration}.pt")
            torch.save({
                "iteration": iteration,
                "model_state_dict": agent.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": env_cfg,
                "avg_reward": avg_reward,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Check if simulation app is still running
        if not simulation_app.is_running():
            break
    
    print("Training complete!")
    
    # Save final model
    final_path = os.path.join(args_cli.output_dir, "final_model.pt")
    torch.save({
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": env_cfg,
    }, final_path)
    print(f"Saved final model to {final_path}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()

