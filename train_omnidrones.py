#!/usr/bin/env python3
"""
OmniDrones PPO Training Script for Drone Navigation

This script trains a PPO agent to navigate a drone to target positions using
the OmniDrones platform built on NVIDIA Isaac Sim.

Features:
- GPU-parallelized simulation (1000s of environments)
- Hydra configuration management
- WandB experiment tracking
- Built-in PPO with TorchRL

Usage:
    # Train with custom DroneToTarget task
    python train_omnidrones.py headless=true task=DroneToTarget
    
    # Train with built-in Hover task (for testing)
    python train_omnidrones.py headless=true task=Hover
    
    # With evaluation and checkpointing
    python train_omnidrones.py headless=true eval_interval=100 save_interval=500
    
    # Enable WandB logging
    python train_omnidrones.py wandb.mode=online wandb.entity=YOUR_USERNAME

Requirements:
    - OmniDrones (pip install -e OmniDrones/)
    - Isaac Lab
    - Isaac Sim 4.1.0
"""

import os
import sys
import logging
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from setproctitle import setproctitle

# OmniDrones imports
from omni_drones import init_simulation_app
from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.learning import ALGOS
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats, RenderCallback
from omni_drones.utils.wandb import init_wandb

# TorchRL imports
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose
from torchrl.envs.utils import set_exploration_type, ExplorationType

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="train")
def main(cfg):
    """Main training function"""
    
    # Resolve Hydra config
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    
    # Initialize simulation
    logger.info("Initializing Isaac Sim simulation...")
    simulation_app = init_simulation_app(cfg)
    
    # Initialize WandB
    run = init_wandb(cfg)
    setproctitle(run.name)
    
    print("=" * 60)
    print("OmniDrones PPO Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    # Get environment class
    try:
        env_class = IsaacEnv.REGISTRY[cfg.task.name]
    except KeyError:
        logger.error(f"Task '{cfg.task.name}' not found in registry.")
        logger.info(f"Available tasks: {list(IsaacEnv.REGISTRY.keys())}")
        simulation_app.close()
        sys.exit(1)
    
    # Create base environment
    logger.info(f"Creating environment: {cfg.task.name}")
    base_env = env_class(cfg, headless=cfg.headless)
    
    # Apply transforms
    transforms = [InitTracker()]
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)
    
    logger.info(f"Environment created with {env.num_envs} parallel environments")
    logger.info(f"Observation spec: {env.observation_spec}")
    logger.info(f"Action spec: {env.action_spec}")
    
    # Initialize policy
    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )
    except KeyError:
        logger.error(f"Algorithm '{cfg.algo.name}' not found.")
        logger.info(f"Available algorithms: {list(ALGOS.keys())}")
        simulation_app.close()
        sys.exit(1)
    
    logger.info(f"Policy initialized: {cfg.algo.name}")
    
    # Training setup
    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    
    # Create data collector
    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )
    
    # Evaluation function
    @torch.no_grad()
    def evaluate(seed: int = 0, exploration_type: ExplorationType = ExplorationType.MODE):
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)
        
        render_callback = RenderCallback(interval=2)
        
        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        
        base_env.enable_render(not cfg.headless)
        env.reset()
        
        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()
        
        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape + (1,) * (tensor.ndim - 2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)
        
        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }
        
        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }
        
        # Log video
        try:
            import wandb
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )
        except Exception as e:
            logger.warning(f"Could not create video: {e}")
        
        return info
    
    # Training loop
    logger.info(f"Starting training for {total_frames:,} frames...")
    pbar = tqdm(collector, total=total_frames // frames_per_batch)
    env.train()
    
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())
        
        # Log episode stats
        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        
        # Training step
        info.update(policy.train_op(data.to_tensordict()))
        
        # Evaluation
        if eval_interval > 0 and i % eval_interval == 0:
            logger.info(f"Evaluation at {collector._frames:,} frames")
            info.update(evaluate())
            env.train()
            base_env.train()
        
        # Save checkpoint
        if save_interval > 0 and i % save_interval == 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")
            except AttributeError:
                logger.warning("Policy does not support state_dict()")
        
        # Log to WandB
        run.log(info)
        
        # Update progress bar
        pbar.set_postfix({
            "fps": f"{collector._fps:.0f}",
            "frames": f"{collector._frames:,}"
        })
        
        # Check max iterations
        if max_iters > 0 and i >= max_iters - 1:
            break
    
    # Final evaluation
    logger.info(f"Final evaluation at {collector._frames:,} frames")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)
    
    # Save final model
    try:
        final_path = "drone_ppo_final.pt"
        torch.save(policy.state_dict(), final_path)
        logger.info(f"Saved final model: {final_path}")
        
        # Also save to WandB
        import wandb
        wandb.save(final_path)
    except Exception as e:
        logger.warning(f"Could not save final model: {e}")
    
    print("=" * 60)
    print("Training Complete!")
    print(f"Total frames: {collector._frames:,}")
    print("=" * 60)
    
    wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
