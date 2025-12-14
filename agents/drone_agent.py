"""
RL Agent for Drone Tank Task
Combines vision and state information for decision making
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from vision.vision_processor import VisionEncoder, VisionProcessor


class DroneActorCritic(nn.Module):
    """
    Actor-Critic network for drone control.
    Processes both vision (camera) and state (position, velocity) observations.
    """
    
    def __init__(self, 
                 state_dim: int = 13,
                 vision_feature_dim: int = 256,
                 action_dim: int = 4,
                 hidden_dims: list = [512, 512, 256],
                 use_vision: bool = True):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: Dimension of state observations
            vision_feature_dim: Dimension of vision features
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
            use_vision: Whether to use vision inputs
        """
        super().__init__()
        
        self.use_vision = use_vision
        self.state_dim = state_dim
        self.vision_feature_dim = vision_feature_dim
        
        # Vision encoder (if using vision)
        if use_vision:
            self.vision_encoder = VisionEncoder(output_dim=vision_feature_dim)
        else:
            self.vision_feature_dim = 0
        
        # Input dimension: state + vision features
        input_dim = state_dim + self.vision_feature_dim
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ELU(),
            ])
            prev_dim = hidden_dim
        
        self.shared_network = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
        )
        
        # Actor log_std (for continuous actions)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 1),
        )
    
    def forward(self, 
                state: torch.Tensor, 
                images: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State observations [batch_size, state_dim]
            images: Camera images [batch_size, channels, height, width] (optional)
            
        Returns:
            action_mean: Mean of action distribution [batch_size, action_dim]
            action_log_std: Log standard deviation [action_dim]
            value: Value estimate [batch_size, 1]
        """
        # Process vision if available
        if self.use_vision and images is not None:
            vision_features = self.vision_encoder(images)
            # Concatenate state and vision features
            combined_input = torch.cat([state, vision_features], dim=-1)
        else:
            combined_input = state
        
        # Shared feature extraction
        features = self.shared_network(combined_input)
        
        # Actor outputs
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, 
                   state: torch.Tensor, 
                   images: torch.Tensor = None,
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: State observations
            images: Camera images (optional)
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        action_mean, action_log_std, value = self.forward(state, images)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            # Sample from normal distribution
            action_std = torch.exp(action_log_std)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_actions(self,
                        state: torch.Tensor,
                        images: torch.Tensor,
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            state: State observations
            images: Camera images
            actions: Actions to evaluate
            
        Returns:
            log_prob: Log probability of actions
            value: Value estimate
            entropy: Entropy of action distribution
        """
        action_mean, action_log_std, value = self.forward(state, images)
        
        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class DroneAgent:
    """
    High-level agent wrapper for drone control.
    Handles action scaling and environment interaction.
    """
    
    def __init__(self, 
                 model: DroneActorCritic,
                 action_scale: torch.Tensor = None,
                 device: str = "cuda:0"):
        """
        Initialize agent.
        
        Args:
            model: Actor-Critic model
            action_scale: Scaling factors for actions [action_dim]
            device: Device to run on
        """
        self.model = model.to(device)
        self.device = device
        
        if action_scale is None:
            self.action_scale = torch.ones(model.actor_mean[0].out_features, device=device)
        else:
            self.action_scale = action_scale.to(device)
    
    def select_action(self, 
                     state: torch.Tensor,
                     images: torch.Tensor = None,
                     deterministic: bool = False) -> torch.Tensor:
        """
        Select action given observations.
        
        Args:
            state: State observations
            images: Camera images
            deterministic: Whether to use deterministic policy
            
        Returns:
            Scaled action
        """
        self.model.eval()
        with torch.no_grad():
            action, _ = self.model.get_action(state, images, deterministic)
            # Scale actions
            scaled_action = action * self.action_scale
        return scaled_action
    
    def train_step(self, 
                  states: torch.Tensor,
                  images: torch.Tensor,
                  actions: torch.Tensor,
                  returns: torch.Tensor,
                  advantages: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform training step.
        
        Args:
            states: Batch of states
            images: Batch of images
            actions: Batch of actions
            returns: Batch of returns
            advantages: Batch of advantages
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        log_probs, values, entropy = self.model.evaluate_actions(states, images, actions)
        
        # Compute policy loss (PPO-style)
        ratio = torch.exp(log_probs - log_probs.detach())
        policy_loss = -(ratio * advantages).mean()
        
        # Value loss
        value_loss = ((values - returns) ** 2).mean()
        
        # Entropy bonus
        entropy_bonus = entropy.mean()
        
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_bonus,
        }

