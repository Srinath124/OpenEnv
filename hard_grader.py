"""Hard task grader - Multi-objective optimization."""
from graders import HardTaskGrader


def grade(state: dict) -> float:
    """
    HARD TASK: Improve IoU while adding missing class coverage,
    reducing training time, and avoiding instability.
    
    Target: Mean IoU >= 0.65 + Sky IoU >= 0.50 + Compute efficient
    Focus: Multi-objective optimization
    Weights: IoU 40% + Sky 30% + Efficiency 15% + Stability 15%
    """
    return HardTaskGrader.score(state)
