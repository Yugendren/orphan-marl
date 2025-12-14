"""
Utility functions for environment setup and management
"""

import numpy as np
import torch
from typing import Tuple, List, Dict, Any


def create_env_config(task_name: str, 
                     num_envs: int = 4096,
                     **kwargs) -> Dict[str, Any]:
    """
    Create environment configuration dictionary.
    
    Args:
        task_name: Name of the task
        num_envs: Number of parallel environments
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration dictionary
    """
    config = {
        "task": task_name,
        "num_envs": num_envs,
        "sim": {
            "dt": 0.02,
            "use_gpu_pipeline": True,
            "physics_engine": "physx",
        },
        **kwargs
    }
    return config


def randomize_spawn_positions(num_envs: int,
                              drone_range: Tuple[float, float, float] = (0.0, 5.0, 0.0),
                              tank_range: Tuple[float, float, float] = (5.0, 15.0, 0.0),
                              seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random spawn positions for drone and tank.
    
    Args:
        num_envs: Number of environments
        drone_range: (x_min, y_min, z_min) for drone spawn area
        tank_range: (x_min, y_min, z_min) for tank spawn area
        seed: Random seed
        
    Returns:
        Tuple of (drone_positions, tank_positions) arrays
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Random drone positions
    drone_positions = np.array([
        np.random.uniform(-2.0, 2.0, num_envs),  # x
        np.random.uniform(-2.0, 2.0, num_envs),  # y
        np.random.uniform(3.0, 7.0, num_envs),  # z (height)
    ]).T
    
    # Random tank positions
    tank_positions = np.array([
        np.random.uniform(5.0, 15.0, num_envs),  # x
        np.random.uniform(-5.0, 5.0, num_envs),  # y
        np.zeros(num_envs),  # z (ground level)
    ]).T
    
    return drone_positions, tank_positions


def compute_contact_forces(contact_data: Dict[str, Any],
                           drone_handle: int,
                           tank_handle: int,
                           ground_handle: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute contact forces between drone and tank/ground.
    
    Args:
        contact_data: Contact data from gym
        drone_handle: Handle to drone actor
        tank_handle: Handle to tank actor
        ground_handle: Handle to ground (optional)
        
    Returns:
        Tuple of (tank_contacts, ground_contacts) boolean tensors
    """
    # Placeholder - in practice, extract from gym contact data
    # This would involve checking contact pairs
    num_envs = contact_data.get("num_envs", 1)
    tank_contacts = torch.zeros(num_envs, dtype=torch.bool)
    ground_contacts = torch.zeros(num_envs, dtype=torch.bool)
    
    return tank_contacts, ground_contacts


def normalize_observations(obs: torch.Tensor,
                          obs_mean: torch.Tensor = None,
                          obs_std: torch.Tensor = None,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize observations using running statistics.
    
    Args:
        obs: Observations to normalize
        obs_mean: Running mean (optional)
        obs_std: Running standard deviation (optional)
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized observations
    """
    if obs_mean is None:
        obs_mean = obs.mean(dim=0, keepdim=True)
    if obs_std is None:
        obs_std = obs.std(dim=0, keepdim=True) + eps
    
    normalized = (obs - obs_mean) / obs_std
    return normalized


def clip_actions(actions: torch.Tensor,
                action_min: torch.Tensor = None,
                action_max: torch.Tensor = None) -> torch.Tensor:
    """
    Clip actions to valid range.
    
    Args:
        actions: Actions to clip
        action_min: Minimum action values
        action_max: Maximum action values
        
    Returns:
        Clipped actions
    """
    if action_min is not None:
        actions = torch.clamp(actions, min=action_min)
    if action_max is not None:
        actions = torch.clamp(actions, max=action_max)
    
    return actions

