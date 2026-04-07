from typing import Tuple, Dict, Any
from models import Action, Observation, Reward
from simulator import Simulator
from reward_engine import RewardEngine


class Environment:
    """
    OffroadSegNet ML Engineering Debugging Environment.
    
    Agent acts as ML engineer diagnosing and fixing an OffroadSegNet model
    to improve semantic segmentation IoU while respecting compute constraints.
    """
    
    def __init__(self, target_iou: float = 0.50, initial_compute: int = 30, seed: int = 42):
        self.target_iou = target_iou
        self.initial_compute = initial_compute
        self.simulator = Simulator(
            target_iou=target_iou,
            initial_compute=initial_compute,
            seed=seed
        )
        self.reward_engine = RewardEngine(
            target_iou=target_iou,
            initial_compute=initial_compute
        )
        self.episode_steps = 0
        self.max_steps = 50  # Reasonable limit per episode

    def reset(self) -> Observation:
        """Reset environment and return initial observation."""
        self.simulator.reset()
        self.reward_engine.reset()
        self.episode_steps = 0
        return self.simulator.get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Returns:
        - observation: Structured experiment report
        - reward: Multi-component reward signal
        - done: Episode termination flag
        - info: Diagnostic information
        """
        self.episode_steps += 1
        
        # Execute action in simulator
        sim_info = self.simulator.step(action)
        observation = self.simulator.get_observation()
        
        # Compute reward using reward engine
        reward_dict = self.reward_engine.compute_reward(
            observation, action.action, sim_info
        )
        
        reward = Reward(
            value=reward_dict["value"],
            iou_reward=reward_dict["iou_reward"],
            coverage_reward=reward_dict["coverage_reward"],
            efficiency_reward=reward_dict["efficiency_reward"],
            penalty=reward_dict["penalty"]
        )
        
        # Determine if episode is done
        done = False
        
        # Terminal condition 1: Out of compute budget
        if observation.compute_remaining_minutes <= 0:
            done = True
        
        # Terminal condition 2: Explicitly stopped training
        if observation.early_stopped:
            done = True
        
        # Terminal condition 3: Max steps reached
        if self.episode_steps >= self.max_steps:
            done = True
        
        info = {
            "step": self.episode_steps,
            "message": sim_info.get("msg", ""),
            "action": action.action,
            "iou_improvement": sim_info.get("iou_improvement", 0),
        }
        
        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        """
        Get full state dict for grading and evaluation.
        Compatible with grader interface.
        """
        return {
            "epoch": self.simulator.total_epochs_trained,
            "mean_iou": self.simulator.mean_iou,
            "road_iou": self.simulator.road_iou,
            "vegetation_iou": self.simulator.vegetation_iou,
            "sky_iou": self.simulator.sky_iou,
            "overfitting": self.simulator.overfitting_flag,
            "instability": self.simulator.instability_flag,
            "remaining_compute": self.simulator.remaining_compute,
            "train_loss": self.simulator.train_loss,
            "val_loss": self.simulator.val_loss,
            "optimizer": self.simulator.optimizer,
            "learning_rate": self.simulator.learning_rate,
            "batch_size": self.simulator.batch_size,
            "class_balancing": self.simulator.class_balancing,
            "augmentation": self.simulator.augmentation_level,
            "regularization": self.simulator.regularization_strength,
        }
