"""
Reward engine for realistic ML workflow feedback.
Provides smooth, bounded, informative reward signals for agent learning.
All rewards normalized to [-1, 1] range for stability.
"""


class RewardEngine:
    """
    Calculates nuanced rewards reflecting real ML engineering priorities:
    - Steady IoU improvement with diminishing returns near target
    - Class coverage (especially sky class)
    - Efficiency (not wasting compute)
    - Stability and recovery from instability
    - Avoiding overfitting
    
    Reward bounds: [-1.0, 1.0] for stability.
    """
    
    def __init__(self, target_iou: float = 0.50, initial_compute: int = 30):
        self.target_iou = target_iou
        self.initial_compute = initial_compute
        self.previous_mean_iou = 0.0
        self.previous_sky_iou = 0.0
        self.previous_instability = True  # Track instability changes
        self.action_count = 0
    
    def compute_reward(self, observation, action_name: str, info: dict) -> dict:
        """
        Compute multi-component reward for an action.
        
        Returns dict with:
        - value: main reward signal (bounded [-1, 1])
        - iou_reward: improvement in IoU (scaled with diminishing returns)
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
        
        # Component 1: IoU Improvement Reward with diminishing returns
        iou_improvement = observation.mean_iou - self.previous_mean_iou
        
        if iou_improvement > 0:
            # Scale improvement by (1 - current_iou) for diminishing returns near target
            scale_factor = 1.0 - observation.mean_iou
            scaled_improvement = iou_improvement * scale_factor
            reward_components["iou_reward"] = min(0.4, scaled_improvement * 5.0)
        elif iou_improvement < 0:
            # Penalize degradation less harshly
            reward_components["iou_reward"] = max(-0.2, iou_improvement * 2.0)
        
        # Component 2: Class Coverage Reward (Sky class critical)
        sky_improvement = observation.sky_iou - self.previous_sky_iou
        if observation.sky_iou > 0 and sky_improvement > 0:
            # Bonus for learning previously-missed class
            reward_components["coverage_reward"] = min(0.2, sky_improvement * 3.0)
        elif observation.sky_iou > 0.3:
            # Sustained reward for maintaining sky learning
            reward_components["coverage_reward"] = 0.1
        
        # Component 3: Efficiency Reward
        if action_name == "run_training_epoch":
            # Training epochs are expensive - evaluate ROI
            if iou_improvement > 0.01:
                reward_components["efficiency_reward"] = 0.15  # Good ROI
            elif iou_improvement > 0:
                reward_components["efficiency_reward"] = 0.05  # Marginal ROI
            elif iou_improvement == 0:
                # Stagnation penalty: no improvement on expensive action
                reward_components["efficiency_reward"] = -0.08
            else:
                reward_components["efficiency_reward"] = -0.1  # Degradation
        else:
            # Configuration actions are cheap - reduced reward (0.02 not 0.05)
            reward_components["efficiency_reward"] = 0.02
        
        # Component 4: Instability Recovery Reward
        if self.previous_instability and not observation.instability:
            # Agent fixed instability - reward recovery
            reward_components["efficiency_reward"] += 0.1
        
        # Component 5: Penalties
        penalty = 0.0
        
        # Instability is critical failure mode
        if observation.instability:
            penalty -= 0.20
        
        # Overfitting hurts real-world performance
        if observation.overfitting:
            penalty -= 0.15
        
        # Repeated no-op config actions waste steps
        if iou_improvement == 0 and action_name not in ["run_training_epoch", "early_stop_training"]:
            penalty -= 0.05
        
        # Running out of compute is bad
        if observation.compute_remaining_minutes <= 0:
            penalty -= 0.20
        
        reward_components["penalty"] = max(-0.5, penalty)
        
        # Component 6: Success Reward (reaching target)
        success_reward = 0.0
        if observation.mean_iou >= self.target_iou:
            success_reward = 0.3
        
        # Compute total value (sum all components, then clamp)
        total = (
            reward_components["iou_reward"] +
            reward_components["coverage_reward"] +
            reward_components["efficiency_reward"] +
            reward_components["penalty"] +
            success_reward
        )
        
        # Strictly clamp to [-1.0, 1.0]
        reward_components["value"] = max(-1.0, min(1.0, total))
        
        # Update tracking
        self.previous_mean_iou = observation.mean_iou
        self.previous_sky_iou = observation.sky_iou
        self.previous_instability = observation.instability
        
        return reward_components
    
    def reset(self):
        """Reset reward tracking for new episode."""
        self.previous_mean_iou = 0.0
        self.previous_sky_iou = 0.0
        self.previous_instability = True
        self.action_count = 0
