"""Medium task grader - IoU with generalization."""
from graders import MediumTaskGrader


def grade(state: dict) -> float:
    """
    MEDIUM TASK: Improve IoU while preventing overfitting.
    
    Target: Mean IoU >= 0.55 + No overfitting
    Focus: Generalization and stability
    Penalizes: Overfitting (50% penalty)
    """
    return MediumTaskGrader.score(state)
