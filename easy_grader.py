"""Easy task grader - Basic IoU improvement."""
from graders import EasyTaskGrader


def grade(state: dict) -> float:
    """
    EASY TASK: Improve IoU from poor baseline to acceptable level.
    
    Target: Mean IoU >= 0.50
    Focus: Basic debugging and training
    Score: 0.0 (baseline) to 1.0 (target)
    """
    return EasyTaskGrader.score(state)
