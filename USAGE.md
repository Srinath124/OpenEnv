# OffroadSegNet ML Environment - Usage Guide

## Quick Start

### 1. Basic Environment Usage

```python
from environment import Environment
from models import Action

# Create environment with target IoU
env = Environment(target_iou=0.55, initial_compute=30)

# Reset for new episode
obs = env.reset()
print(obs.report)  # Detailed experiment report

# Take action
action = Action(action="adjust_learning_rate")
obs, reward, done, info = env.step(action)

print(f"IoU: {obs.mean_iou:.4f}")
print(f"Reward: {reward.value:.2f}")
print(f"Done: {done}")

# Get final state for grading
state = env.state()
```

### 2. Run a Complete Episode

```python
from environment import Environment
from models import Action
from graders import EasyTaskGrader

def run_episode(strategy_actions):
    """Run predefined sequence of actions."""
    env = Environment()
    obs = env.reset()
    
    print("Initial State:")
    print(f"  Mean IoU: {obs.mean_iou:.4f}")
    print(f"  Compute: {obs.compute_remaining_minutes}m")
    print()
    
    for action_name in strategy_actions:
        obs, reward, done, info = env.step(Action(action=action_name))
        
        print(f"Action: {action_name}")
        print(f"  → IoU: {obs.mean_iou:.4f} | Reward: {reward.value:+.2f}")
        
        if done:
            print("Episode ended")
            break
    
    # Grade performance
    state = env.state()
    score = EasyTaskGrader.score(state)
    
    print(f"\nFinal Score: {score:.4f}/1.0")
    print(f"Final IoU: {obs.mean_iou:.4f}/0.50")
    
    return score

# Optimal strategy for easy task
optimal_actions = [
    "adjust_learning_rate",      # Fix instability
    "enable_augmentation",       # Improve generalization
    "enable_class_balancing",    # Learn sky class
    "run_training_epoch",        # Train
    "run_training_epoch",        # Train
    "run_training_epoch",        # Train
]

score = run_episode(optimal_actions)
```

### 3. Understand the Observation Report

```python
from environment import Environment

env = Environment()
obs = env.reset()
print(obs.report)

# Output:
# ======================================================================
# OFFROAD SEGMENTATION EXPERIMENT REPORT
# ======================================================================
# Model: OffroadSegNet ResNet-18
# Dataset: Offroad Terrain Segmentation (Train + Val)
# Epoch: 0 | Compute Budget: 30m remaining
# 
# CURRENT METRICS:
#   Mean IoU:        0.1667
#   Road IoU:        0.4000
#   Vegetation IoU:  0.1000
#   Sky IoU:         0.0000 (missing coverage)
#   Train Loss:      1.2000
#   Val Loss:        1.5000
# 
# TRAINING CONFIGURATION:
#   Optimizer:             SGD
#   Learning Rate:         0.01
#   Batch Size:            32
#   Backbone Depth:        18
#   Feature Channels:      256
#   L2 Regularization:     0.0
#   Data Augmentation:     Disabled
#   Class Balancing:       Disabled
#   Mixed Precision:       Disabled
# 
# ISSUES DETECTED:
#   ⚠️  Training instability - suboptimal learning rate
#   ⚠️  Sky class not learning - severe class imbalance
# 
# RECOMMENDATIONS:
#   • Target IoU: 0.50 | Gap: 0.3333
```

### 4. Work with Rewards

```python
from environment import Environment
from models import Action

env = Environment()
obs = env.reset()

# Take action and examine reward components
obs, reward, done, info = env.step(Action(action="adjust_learning_rate"))

print(f"Total Reward:     {reward.value:+.2f}")
print(f"  IoU Reward:     {reward.iou_reward:+.2f}")
print(f"  Coverage Reward: {reward.coverage_reward:+.2f}")
print(f"  Efficiency Reward: {reward.efficiency_reward:+.2f}")
print(f"  Penalty:        {reward.penalty:+.2f}")

# Output:
# Total Reward:     +0.30
#   IoU Reward:     +0.00
#   Coverage Reward: +0.00
#   Efficiency Reward: +0.10
#   Penalty:        -0.00
```

### 5. Evaluate on Different Tasks

```python
from environment import Environment
from models import Action
from graders import EasyTaskGrader, MediumTaskGrader, HardTaskGrader

def evaluate_on_task(task_name):
    """Run agent on specific task difficulty."""
    
    targets = {
        "easy": 0.50,
        "medium": 0.55,
        "hard": 0.65,
    }
    
    target = targets[task_name]
    env = Environment(target_iou=target)
    obs = env.reset()
    
    # Example strategy
    actions = [
        "adjust_learning_rate",
        "enable_augmentation",
        "enable_class_balancing",
    ] + ["run_training_epoch"] * 5
    
    for action_name in actions:
        obs, reward, done, info = env.step(Action(action=action_name))
        if done:
            break
    
    # Score on appropriate grader
    state = env.state()
    
    if task_name == "easy":
        score = EasyTaskGrader.score(state)
    elif task_name == "medium":
        score = MediumTaskGrader.score(state)
    else:
        score = HardTaskGrader.score(state)
    
    return score, state

# Run on all difficulties
for task in ["easy", "medium", "hard"]:
    score, state = evaluate_on_task(task)
    print(f"{task.upper():8} → Score: {score:.4f} | IoU: {state['mean_iou']:.4f}")
```

### 6. Using OpenAI for Agent Decisions

```python
import os
from environment import Environment
from models import Action
from inference import MLAgentInference

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-..."

env = Environment()
agent = MLAgentInference(model="gpt-4")

obs = env.reset()

for step in range(15):
    # Get action from GPT-4
    action_name = agent.get_action(obs.report)
    
    if not action_name:
        # Fallback to heuristic
        action_name = agent.heuristic_action(obs.report)
    
    print(f"[Step {step}] Action: {action_name}")
    obs, reward, done, info = env.step(Action(action=action_name))
    
    if done:
        break

state = env.state()
print(f"Final IoU: {state['mean_iou']:.4f}")
```

### 7. Testing & Validation

```python
# Run built-in tests
python test_environment.py

# Output:
# test_environment_crash_free ✓
# test_random_agent ✓
# test_heuristic_agent ✓
# test_reward_monotonicity ✓
# test_environment_stability ✓
# test_long_episode ✓
# test_grader_consistency ✓
# test_action_validation ✓
# test_observation_report_format ✓
```

### 8. Comprehensive Evaluation

```python
# Run full evaluation suite
python evaluation.py

# Output:
# ======================================================================
# OFFROAD SEGMENTATION ENVIRONMENT EVALUATION
# ======================================================================
# 
# Evaluating EASY Task (Random Agent)...
#   Success Rate:          0.0%
#   Average Task Score:    0.3595/1.0
#   Avg IoU Improvement:   0.0131
#   Avg Steps to Complete: 7.0
#   Seed Scores:           ['0.360', '0.360', '0.360']
# 
# [MEDIUM and HARD similarly...]
# 
# ======================================================================
# SUMMARY
# ======================================================================
# Overall Average Score:   0.3358/1.0
# Easy Task:               0.3595/1.0
# Medium Task:             0.3268/1.0
# Hard Task:               0.3213/1.0
# Total Episodes Run:      9
# ======================================================================
```

### 9. Advanced: Custom Strategy Implementation

```python
from environment import Environment
from models import Action
from graders import HardTaskGrader

class MLEngineeringStrategy:
    """Example strategy for hard task."""
    
    def __init__(self):
        self.configuration_phase_complete = False
    
    def decide(self, observation):
        """Decide next action based on observation state."""
        
        # Phase 1: Fix critical issues
        if observation.instability and not self.configuration_phase_complete:
            return "adjust_learning_rate"
        
        if observation.sky_iou == 0.0 and not self.configuration_phase_complete:
            return "enable_class_balancing"
        
        # Phase 2: Improve generalization
        if not self.configuration_phase_complete:
            self.configuration_phase_complete = True
            return "enable_augmentation"
        
        # Phase 3: Train
        if observation.compute_remaining_minutes > 10:
            return "run_training_epoch"
        else:
            return "early_stop_training"


def run_strategic_episode():
    env = Environment(target_iou=0.65)  # Hard task
    strategy = MLEngineeringStrategy()
    
    obs = env.reset()
    total_reward = 0.0
    
    for step in range(50):
        action_name = strategy.decide(obs)
        obs, reward, done, info = env.step(Action(action=action_name))
        
        total_reward += reward.value
        
        if done:
            break
    
    state = env.state()
    score = HardTaskGrader.score(state)
    
    print(f"Strategy Performance:")
    print(f"  Score: {score:.4f}/1.0")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Final IoU: {state['mean_iou']:.4f}")
    print(f"  Sky IoU: {state['sky_iou']:.4f}")
    print(f"  Stability: {'✓' if not state['instability'] else '✗'}")
    print(f"  Overfitting: {'✗' if not state['overfitting'] else '✓ problematic'}")

run_strategic_episode()
```

## Action Reference

### Configuration Actions (no compute cost)

| Action | Effect | When to Use |
|--------|--------|------------|
| `adjust_learning_rate` | Fix 0.01 → 0.001, resolve instability | Immediately if instability flag is true |
| `enable_augmentation` | +10% generalization, slight GPU overhead | Before intensive training |
| `increase_regularization` | Combat overfitting via L2 norm | When val loss diverging from train loss |
| `switch_optimizer` | SGD → AdamW, +15% convergence | Anytime, generally helpful |
| `enable_class_balancing` | Focal loss, enables sky learning | When sky_iou stuck at 0.0 |
| `reduce_batch_size` | 32 → 16, memory efficient | If memory constraints (not enforced here) |

### Training Actions (costs compute)

| Action | Cost | Effect | When to Use |
|--------|------|--------|------------|
| `run_training_epoch` | 5-6 min | Train one epoch | When configuration ready |
| `early_stop_training` | - | Stop training, freeze state | When done or budget critical |

## Common Patterns

### Solving Easy Task
```
adjust_learning_rate → run_training_epoch × 4
Expected: 0.90+ score
```

### Solving Medium Task
```
adjust_learning_rate → enable_augmentation → increase_regularization 
→ run_training_epoch × 4
Expected: 0.85+ score
```

### Solving Hard Task
```
adjust_learning_rate → enable_class_balancing → enable_augmentation 
→ increase_regularization → run_training_epoch × 5
Expected: 0.80+ score
```

## Troubleshooting

**Q: Why isn't sky IoU improving?**
A: Enable class balancing. Without it, sky class receives no gradient signal.

**Q: Why is training unstable?**
A: Instability flag is true - use `adjust_learning_rate`.

**Q: Why is IoU decreasing?**
A: If instability is true, training causes degradation. Fix it first.

**Q: Why did I run out of compute?**
A: Each epoch costs 5-6 minutes. Plan: 30 minutes ÷ 5 = ~6 epochs max.

**Q: How do I prevent overfitting?**
A: Use `increase_regularization` and `enable_augmentation`.

## Performance Benchmarks

**Random Agent (3 seeds):**
- Easy: 0.36/1.0
- Medium: 0.33/1.0
- Hard: 0.32/1.0

**Heuristic Agent:**
- Easy: 0.90+/1.0
- Medium: 0.70+/1.0
- Hard: 0.60+/1.0

**Expert Agent (optimal configuration):**
- Easy: 1.00/1.0
- Medium: 0.95+/1.0
- Hard: 0.85+/1.0

## References

- `ARCHITECTURE.md` - Detailed design documentation
-` models.py` - Pydantic schema definitions
- `graders.py` - Task grading logic
- `test_environment.py` - Implementation examples
- `openenv.yaml` - Environment metadata
