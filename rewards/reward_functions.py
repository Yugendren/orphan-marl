"""
Reward functions for Drone Tank Task
"""

import torch
from typing import Dict, Any


class RewardFunction:
    """Base class for reward functions."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def compute(self, **kwargs) -> torch.Tensor:
        """Compute reward."""
        raise NotImplementedError


class TankHitReward(RewardFunction):
    """Positive reward when drone hits/touches the tank."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.reward_value = config.get("reward_value", 10.0)
        self.contact_threshold = config.get("contact_threshold", 0.5)
        self.use_distance = config.get("use_distance", False)
    
    def compute(self, drone_pos: torch.Tensor, tank_pos: torch.Tensor, 
                distance: torch.Tensor = None) -> torch.Tensor:
        """
        Compute reward for hitting the tank.
        
        Args:
            drone_pos: Drone positions [num_envs, 3]
            tank_pos: Tank positions [num_envs, 3]
            distance: Pre-computed distances [num_envs] (optional)
            
        Returns:
            Reward tensor [num_envs]
        """
        if distance is None:
            distance = torch.norm(drone_pos - tank_pos, dim=-1)
        
        # Binary reward: full reward if within threshold
        hit_mask = (distance < self.contact_threshold).float()
        reward = hit_mask * self.reward_value
        
        # Optional: distance-based reward (smooth reward as getting closer)
        if self.use_distance:
            distance_reward = torch.exp(-distance / self.contact_threshold) * self.reward_value
            reward = torch.maximum(reward, distance_reward)
        
        return reward


class GroundCrashPenalty(RewardFunction):
    """Negative reward (penalty) when drone crashes into the ground."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.penalty_value = config.get("penalty_value", -5.0)
        self.contact_threshold = config.get("contact_threshold", 0.3)
        self.ground_height = config.get("ground_height", 0.0)
    
    def compute(self, drone_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for crashing into the ground.
        
        Args:
            drone_pos: Drone positions [num_envs, 3]
            
        Returns:
            Penalty tensor [num_envs] (negative values)
        """
        height = drone_pos[:, 2]
        distance_to_ground = torch.abs(height - self.ground_height)
        
        # Penalty if too close to ground
        crash_mask = (distance_to_ground < self.contact_threshold).float()
        penalty = crash_mask * self.penalty_value
        
        return penalty


class DistanceReward(RewardFunction):
    """Reward based on distance to target (encourages getting closer)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scale = config.get("scale", 0.1)
        self.max_distance = config.get("max_distance", 20.0)
    
    def compute(self, drone_pos: torch.Tensor, tank_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute distance-based reward.
        
        Args:
            drone_pos: Drone positions [num_envs, 3]
            tank_pos: Tank positions [num_envs, 3]
            
        Returns:
            Reward tensor [num_envs]
        """
        distance = torch.norm(drone_pos - tank_pos, dim=-1)
        
        # Negative reward proportional to distance (closer = better)
        reward = -distance * self.scale
        
        # Normalize by max distance
        reward = reward / self.max_distance
        
        return reward


class SurvivalReward(RewardFunction):
    """Small positive reward for staying alive (encourages exploration)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.reward_value = config.get("reward_value", 0.01)
    
    def compute(self, num_envs: int) -> torch.Tensor:
        """
        Compute survival reward.
        
        Args:
            num_envs: Number of environments
            
        Returns:
            Reward tensor [num_envs]
        """
        return torch.ones(num_envs, device=self.config.get("device", "cuda:0")) * self.reward_value


class CombinedReward:
    """Combines multiple reward functions."""
    
    def __init__(self, reward_functions: Dict[str, RewardFunction], weights: Dict[str, float] = None):
        """
        Initialize combined reward function.
        
        Args:
            reward_functions: Dictionary of reward function name -> RewardFunction
            weights: Optional weights for each reward function
        """
        self.reward_functions = reward_functions
        self.weights = weights or {name: 1.0 for name in reward_functions.keys()}
    
    def compute(self, **kwargs) -> torch.Tensor:
        """
        Compute combined reward.
        
        Args:
            **kwargs: Arguments to pass to reward functions
            
        Returns:
            Combined reward tensor [num_envs]
        """
        total_reward = None
        
        for name, reward_fn in self.reward_functions.items():
            reward = reward_fn.compute(**kwargs)
            weight = self.weights.get(name, 1.0)
            weighted_reward = reward * weight
            
            if total_reward is None:
                total_reward = weighted_reward
            else:
                total_reward += weighted_reward
        
        return total_reward

