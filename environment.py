from typing import Tuple, Dict, Any, Literal
from models import Action, Observation, Reward
from simulator import Simulator
from reward_engine import RewardEngine


class TerminationEngine:
    """Encapsulates termination logic."""
    
    @staticmethod
    def check_termination(
        episode_steps: int,
        max_steps: int,
        compute_remaining: int,
        mean_iou: float,
        target_iou: float,
        early_stopped: bool
    ) -> Tuple[bool, Literal["success", "target_reached", "compute_exhausted", "max_steps", "early_stop"]]:
        """
        Check termination conditions and return (done, reason).
        
        Returns:
            done: True if episode should terminate
            reason: Termination reason string
        """
        # Priority: explicit success first
        if mean_iou >= target_iou:
            return True, "target_reached"
        
        # Early stop by agent
        if early_stopped:
            return True, "early_stop"
        
        # Compute budget exhausted
        if compute_remaining <= 0:
            return True, "compute_exhausted"
        
        # Max steps reached
        if episode_steps >= max_steps:
            return True, "max_steps"
        
        return False, ""


class ActionValidator:
    """Validates actions for correctness and realism."""
    
    VALID_ACTIONS = {
        "adjust_learning_rate",
        "enable_augmentation",
        "increase_regularization",
        "switch_optimizer",
        "enable_class_balancing",
        "reduce_batch_size",
        "run_training_epoch",
        "early_stop_training"
    }
    
    # Configuration actions (not duplicable)
    CONFIG_ACTIONS = {
        "adjust_learning_rate",
        "enable_augmentation",
        "increase_regularization",
        "switch_optimizer",
        "enable_class_balancing",
        "reduce_batch_size"
    }
    
    @staticmethod
    def validate(action: str) -> Tuple[bool, str]:
        """
        Validate action.
        Returns (is_valid, reason_if_invalid)
        """
        if action not in ActionValidator.VALID_ACTIONS:
            return False, f"Invalid action: {action}"
        return True, ""
    
    @staticmethod
    def is_config_action(action: str) -> bool:
        """Check if action is a configuration change (shouldn't repeat immediately)."""
        return action in ActionValidator.CONFIG_ACTIONS


class Environment:
    """
    OffroadSegNet ML Engineering Debugging Environment.
    
    Agent acts as ML engineer diagnosing and fixing an OffroadSegNet model
    to improve semantic segmentation IoU while respecting compute constraints.
    
    OpenEnv-compliant with explicit termination, action validation, and debug mode.
    """
    
    def __init__(
        self,
        target_iou: float = 0.50,
        initial_compute: int = 30,
        seed: int = 42,
        debug: bool = False,
        max_steps: int = None
    ):
        self.target_iou = target_iou
        self.initial_compute = initial_compute
        self.seed = seed
        self.debug = debug
        # Link max_steps to compute: initial_compute * 2
        self.max_steps = max_steps if max_steps is not None else (initial_compute * 2)
        
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
        self.action_history: list[str] = []
        self.termination_reason: Literal["success", "target_reached", "compute_exhausted", "max_steps", "early_stop"] = ""
        self.episode_total_reward = 0.0
    
    def reset(self, seed: int = None) -> Observation:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.seed = seed
            self.simulator.seed = seed
            self.simulator.reset(seed=seed)
        else:
            self.simulator.reset()
        
        self.reward_engine.reset()
        self.episode_steps = 0
        self.action_history = []
        self.termination_reason = ""
        self.episode_total_reward = 0.0
        
        return self.simulator.get_observation()
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one step in the environment (OpenEnv compliant).
        Always returns valid observation, reward, done, info.
        Handles invalid actions gracefully with penalty.
        
        Args:
            action: Action object with action field
            
        Returns:
            observation: Current state (always safe)
            reward: Multi-component reward signal (always bounded [-1, 1])
            done: Episode termination flag (boolean)
            info: Diagnostic information (includes success, termination_reason, mean_iou, remaining_compute)
        """
        self.episode_steps += 1
        
        # Validate action
        is_valid, error_msg = ActionValidator.validate(action.action)
        if not is_valid:
            # Invalid action: apply penalty, return safe observation
            observation = self.simulator.get_observation()
            reward = Reward(
                value=-0.2,
                iou_reward=0.0,
                coverage_reward=0.0,
                efficiency_reward=-0.2,
                penalty=-0.2
            )
            self.episode_total_reward += reward.value
            info = {
                "step": self.episode_steps,
                "message": f"Invalid action: {action.action}",
                "action": action.action,
                "iou_improvement": 0.0,
                "repeating_penalty": 0.0,
                "termination_reason": None,
                "success": observation.mean_iou >= self.target_iou,
                "mean_iou": observation.mean_iou,
                "remaining_compute": observation.compute_remaining_minutes,
                "episode_total_reward": self.episode_total_reward
            }
            return observation, reward, False, info
        
        # Track action history
        self.action_history.append(action.action)
        
        # Execute action in simulator
        sim_info = self.simulator.step(action)
        observation = self.simulator.get_observation()
        
        # Compute reward using reward engine
        reward_dict = self.reward_engine.compute_reward(
            observation, action.action, sim_info
        )
        
        # Add terminal success bonus if target reached (bounded)
        if observation.mean_iou >= self.target_iou:
            reward_dict["value"] = min(1.0, reward_dict["value"] + 0.5)  # Capped success bonus
        
        # Ensure reward strictly bounded [-1, 1]
        reward_value = max(-1.0, min(1.0, reward_dict["value"]))
        
        reward = Reward(
            value=reward_value,
            iou_reward=reward_dict["iou_reward"],
            coverage_reward=reward_dict["coverage_reward"],
            efficiency_reward=reward_dict["efficiency_reward"],
            penalty=reward_dict["penalty"]
        )
        
        self.episode_total_reward += reward.value
        
        # Check termination conditions
        done, reason = TerminationEngine.check_termination(
            episode_steps=self.episode_steps,
            max_steps=self.max_steps,
            compute_remaining=observation.compute_remaining_minutes,
            mean_iou=observation.mean_iou,
            target_iou=self.target_iou,
            early_stopped=observation.early_stopped
        )
        
        if done:
            self.termination_reason = reason
        
        # Debug output
        if self.debug:
            self._print_debug_info(action, reward, done, reason if done else "")
        
        # Determine success (reached target IoU)
        success = observation.mean_iou >= self.target_iou
        
        # Build complete info dict (always include required fields)
        info = {
            "step": self.episode_steps,
            "message": sim_info.get("msg", ""),
            "action": action.action,
            "iou_improvement": sim_info.get("iou_improvement", 0.0),
            "repeating_penalty": sim_info.get("repeating_penalty", 0.0),
            "termination_reason": self.termination_reason if done else None,
            "success": success,
            "mean_iou": observation.mean_iou,
            "remaining_compute": observation.compute_remaining_minutes,
            "episode_total_reward": self.episode_total_reward
        }
        
        return observation, reward, done, info
    
    def state(self) -> Dict[str, Any]:
        """
        Get full state dict for grading and evaluation (OpenEnv compatible).
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
            "termination_reason": self.termination_reason,
            "total_reward": self.episode_total_reward
        }
    
    def _print_debug_info(self, action: Action, reward: Reward, done: bool, reason: str) -> None:
        """Print debug information about step execution."""
        print(f"\n[DEBUG Step {self.episode_steps}]")
        print(f"  Action: {action.action}")
        print(f"  Reward breakdown:")
        print(f"    - IoU reward:      {reward.iou_reward:+.3f}")
        print(f"    - Coverage reward: {reward.coverage_reward:+.3f}")
        print(f"    - Efficiency:      {reward.efficiency_reward:+.3f}")
        print(f"    - Penalty:         {reward.penalty:+.3f}")
        print(f"    - Total:           {reward.value:+.3f}")
        print(f"  Metrics: Mean IoU={self.simulator.mean_iou:.4f}, "
              f"Remaining compute={self.simulator.remaining_compute}, "
              f"Epochs={self.simulator.total_epochs_trained}")
        if done:
            print(f"  EPISODE DONE: {reason}")

