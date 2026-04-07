"""
Pydantic models for OffroadSegNet ML environment.
Defines OpenEnv-compliant schema for observations, actions, and rewards.
"""

from pydantic import BaseModel, Field
from typing import Literal


# Action Schema
ActionType = Literal[
    "adjust_learning_rate",
    "enable_augmentation",
    "increase_regularization",
    "switch_optimizer",
    "enable_class_balancing",
    "reduce_batch_size",
    "run_training_epoch",
    "early_stop_training"
]


class Action(BaseModel):
    """Agent action in ML engineering workflow."""
    action: ActionType = Field(
        ..., 
        description="The ML engineering action to perform on model configuration or training."
    )


# Reward Schema
class Reward(BaseModel):
    """Multi-component reward signal for agent learning."""
    value: float = Field(
        0.0, 
        description="Total reward signal (sum of components minus penalties)"
    )
    iou_reward: float = Field(
        0.0, 
        description="Reward component: IoU improvement"
    )
    coverage_reward: float = Field(
        0.0, 
        description="Reward component: New class coverage (e.g., learning sky class)"
    )
    efficiency_reward: float = Field(
        0.0, 
        description="Reward component: Thoughtful compute budget usage"
    )
    penalty: float = Field(
        0.0, 
        description="Penalty component: Instability, overfitting, wasted actions"
    )


# Observation Schema
class Observation(BaseModel):
    """
    Structured experiment report observation.
    Contains metrics and diagnostics for ML engineer agent.
    """
    report: str = Field(
        ..., 
        description="Formatted experiment report with metrics, configuration, and issues detected"
    )
    mean_iou: float = Field(
        ..., 
        description="Mean Intersection over Union across all semantic classes (0-1)"
    )
    road_iou: float = Field(
        ..., 
        description="IoU for road class (critical for off-road navigation)"
    )
    vegetation_iou: float = Field(
        ..., 
        description="IoU for vegetation class"
    )
    sky_iou: float = Field(
        ..., 
        description="IoU for sky class (challenging minority class)"
    )
    epoch: int = Field(
        ..., 
        description="Current training epoch number"
    )
    compute_remaining_minutes: int = Field(
        ..., 
        description="Remaining compute budget in minutes (decreases with training)"
    )
    overfitting: bool = Field(
        ..., 
        description="Whether model is currently overfitting (val loss > train loss)"
    )
    instability: bool = Field(
        ..., 
        description="Whether training is unstable (suboptimal learning rate)"
    )
    early_stopped: bool = Field(
        False, 
        description="Whether training was explicitly stopped by early_stop_training action"
    )

