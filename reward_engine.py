"""
Reward engine for realistic ML workflow feedback.
Provides smooth, informative reward signals for agent learning.
"""


class RewardEngine:
    """
    Calculates nuanced rewards reflecting real ML engineering priorities:
    - Steady IoU improvement
    - Class coverage
    - Efficiency (not wasting compute)
    - Stability (avoiding instability)
    - Avoiding overfitting
    """
    
    def __init__(self, target_iou: float = 0.50, initial_compute: int = 30):
        self.target_iou = target_iou
        self.initial_compute = initial_compute
        self.previous_mean_iou = 0.0
        self.previous_sky_iou = 0.0
        self.action_count = 0
    
    def compute_reward(self, observation, action_name: str, info: dict) -> dict:
        """
        Compute multi-component reward for an action.
        
        Returns dict with:
        - value: main reward signal
        - iou_reward: improvement in IoU
        - coverage_reward: new class coverage
        - efficiency_reward: compute budget utilization  
        - penalty: penalties for bad actions
        """
        self.action_count += 1
        reward_components = {
            "value": 0.0,
            "iou_reward": 0.0,
            "coverage_reward": 0.0,
            "efficiency_reward": 0.0,
            "penalty": 0.0,
        }
        
        # Component 1: IoU Improvement Reward
        iou_improvement = observation.mean_iou - self.previous_mean_iou
        if iou_improvement > 0:
            # Reward improvement proportionally
            reward_components["iou_reward"] = iou_improvement * 10.0
        elif iou_improvement < 0:
            # Penalize degradation
            reward_components["iou_reward"] = iou_improvement * 5.0
        
        # Component 2: Class Coverage Reward
        sky_improvement = observation.sky_iou - self.previous_sky_iou
        if observation.sky_iou > 0:
            # Successfully learning previously-ignored class is valuable
            reward_components["coverage_reward"] = observation.sky_iou * 2.0
            if sky_improvement > 0:
                reward_components["coverage_reward"] += sky_improvement * 5.0
        
        # Component 3: Efficiency Reward
        # Reward agents for being thoughtful about compute usage
        compute_used = self.initial_compute - observation.compute_remaining_minutes
        compute_efficiency = (observation.compute_remaining_minutes / self.initial_compute) if self.initial_compute > 0 else 0
        
        if action_name == "run_training_epoch":
            # Training epochs are expensive but necessary - evaluate cost-benefit
            if iou_improvement > 0.01:
                # Good return on investment
                reward_components["efficiency_reward"] = 0.5
            else:
                # Marginal or negative returns
                reward_components["efficiency_reward"] = -0.2
        else:
            # Configuration actions are cheap - slight reward for using them
            reward_components["efficiency_reward"] = 0.1
        
        # Component 4: Penalties
        # Instability is a major failure mode in ML
        if observation.instability:
            reward_components["penalty"] -= 1.0
        
        # Overfitting hurts real-world performance
        if observation.overfitting:
            reward_components["penalty"] -= 0.5
        
        # Repeated configuration actions waste agent steps
        if info.get("iou_improvement", 0) == 0 and action_name != "run_training_epoch":
            reward_components["penalty"] -= 0.2
        
        # Running out of compute is terminal failure
        if observation.compute_remaining_minutes <= 0:
            reward_components["penalty"] -= 2.0
        
        # Compute total value
        reward_components["value"] = (
            reward_components["iou_reward"] +
            reward_components["coverage_reward"] +
            reward_components["efficiency_reward"] -
            reward_components["penalty"]
        )
        
        # Bound value to reasonable range
        reward_components["value"] = max(-5.0, min(5.0, reward_components["value"]))
        
        # Update tracking
        self.previous_mean_iou = observation.mean_iou
        self.previous_sky_iou = observation.sky_iou
        
        return reward_components
    
    def reset(self):
        """Reset reward tracking for new episode."""
        self.previous_mean_iou = 0.0
        self.previous_sky_iou = 0.0
        self.action_count = 0
