"""
Task graders for OffroadSegNet ML engineering environment.
Each grader evaluates performance on a specific difficulty level.
"""


class EasyTaskGrader:
    """
    EASY TASK: Improve IoU from poor baseline to acceptable level.
    Target: Reach Mean IoU >= 0.50
    Focus: Basic debugging and training
    Score: 0.0 (baseline) to 1.0 (target met)
    """
    
    TARGET_IOU = 0.50
    
    @staticmethod
    def score(state: dict) -> float:
        """
        Score based on IoU improvement.
        Linear scaling from 0 (baseline) to 1 (target).
        Small penalty if instability present.
        """
        mean_iou = state.get("mean_iou", 0.0)
        instability = state.get("instability", False)
        
        if mean_iou >= EasyTaskGrader.TARGET_IOU:
            base_score = 1.0
        else:
            # Linear progression: 0 IoU -> 0 score, 0.50 IoU -> 1.0 score
            base_score = max(0.0, mean_iou / EasyTaskGrader.TARGET_IOU)
        
        # Small penalty if unstable
        if instability:
            base_score *= 0.9
        
        return max(0.0, min(1.0, base_score))


class MediumTaskGrader:
    """
    MEDIUM TASK: Improve IoU while preventing overfitting.
    Target: Reach Mean IoU >= 0.55 without overfitting
    Focus: Generalization and stability
    Penalizes: Overfitting with reduced multiplier (more forgiving)
    """
    
    TARGET_IOU = 0.55
    
    @staticmethod
    def score(state: dict) -> float:
        """
        Score based on IoU + generalization.
        Overfitting causes 30% score reduction (less harsh than before).
        """
        mean_iou = state.get("mean_iou", 0.0)
        overfitting = state.get("overfitting", False)
        
        # Base score from IoU
        base_score = min(1.0, mean_iou / MediumTaskGrader.TARGET_IOU)
        
        # Reduced penalty for overfitting (0.7 multiplier instead of 0.5)
        if overfitting:
            base_score *= 0.7
        
        return max(0.0, min(1.0, base_score))


class HardTaskGrader:
    """
    HARD TASK: Improve IoU while adding missing class coverage,
    reducing training time, and avoiding instability.
    
    Target: Mean IoU >= 0.65 + Sky IoU >= 0.50 + Compute efficient
    Focus: Multi-objective optimization
    Balances: Performance, coverage, efficiency, stability
    """
    
    TARGET_MEAN_IOU = 0.65
    TARGET_SKY_IOU = 0.50
    INITIAL_COMPUTE = 30  # minutes
    
    @staticmethod
    def score(state: dict) -> float:
        """
        Composite score balancing multiple objectives.
        
        Weights:
        - IoU progress: 40% (target 0.65)
        - Sky class coverage: 30% (target 0.50)
        - Efficiency: 15% (compute ratio based)
        - Stability: 15% (allow partial credit)
        """
        mean_iou = state.get("mean_iou", 0.0)
        sky_iou = state.get("sky_iou", 0.0)
        instability = state.get("instability", False)
        compute_remaining = state.get("remaining_compute", 0)
        
        score = 0.0
        
        # Component 1: IoU progress (40% weight)
        iou_component = min(1.0, mean_iou / HardTaskGrader.TARGET_MEAN_IOU)
        score += iou_component * 0.40
        
        # Component 2: Sky class coverage (30% weight)
        # Sky class is hardest to learn - critical for complete understanding
        sky_component = min(1.0, sky_iou / HardTaskGrader.TARGET_SKY_IOU)
        score += sky_component * 0.30
        
        # Component 3: Efficiency (15% weight)
        # Use compute ratio instead of binary threshold
        efficiency_ratio = compute_remaining / HardTaskGrader.INITIAL_COMPUTE
        efficiency_component = max(0.0, efficiency_ratio)  # 1.0 if no compute used, scales down
        score += efficiency_component * 0.15
        
        # Component 4: Stability (15% weight)
        # Allow partial credit - instability doesn't completely eliminate score
        stability_component = 0.3 if instability else 1.0  # 30% if unstable, 100% if stable
        score += stability_component * 0.15
        
        return min(1.0, max(0.0, score))


def grade_easy(state: dict) -> float:
    """Backward compatibility: easy grading function."""
    return EasyTaskGrader.score(state)


def grade_medium(state: dict) -> float:
    """Backward compatibility: medium grading function."""
    return MediumTaskGrader.score(state)


def grade_hard(state: dict) -> float:
    """Backward compatibility: hard grading function."""
    return HardTaskGrader.score(state)
