# OffroadSegNet ML Environment - Quick Reference

## Environment Overview

```
┌─────────────────────────────────────────────────────────────────┐
│          OffroadSegNet ML Debugging Environment v2.0            │
│                                                                 │
│  Agent = ML Engineer debugging semantic segmentation model      │
│  Goal = Improve IoU while respecting compute budget             │
│  Constraint = 30 minutes compute budget maximum                 │
└─────────────────────────────────────────────────────────────────┘
```

## Baseline State (Every New Episode)

| Metric | Value | Note |
|--------|-------|------|
| Road IoU | 0.40 | Partially learned |
| Vegetation IoU | 0.10 | Underlearned |
| Sky IoU | 0.00 | **Missing completely** |
| Mean IoU | 0.167 | Target: 0.50 (easy) |
| Learning Rate | 0.01 | **UNSTABLE** |
| Instability Flag | True | Must fix first |
| Optimizer | SGD | Can switch to AdamW |
| Augmentation | Disabled | Improves by +10% |
| Class Balancing | Disabled | **Critical for sky** |
| Compute Budget | 30 min | Finite resource |

## Actions Available (8 total)

### Configuration (No Cost) — Use Early
```
adjust_learning_rate      → Fix instability (CRITICAL first step)
enable_augmentation       → +10% generalization improvement
increase_regularization   → Combat overfitting
switch_optimizer          → SGD → AdamW (+15% convergence)
enable_class_balancing    → Enables sky class learning
reduce_batch_size         → Memory efficiency (slower training)
```

### Training (Cost: 5-6 min)
```
run_training_epoch        → Train one epoch
                           Cost: 5 min (std), 6 min (small batch)
early_stop_training       → Preserve remaining budget
```

## Tasks & Scoring

### EASY ⭐
| Target | Grading | Success | Optimal Path |
|--------|---------|---------|--------------|
| Mean IoU ≥ 0.50 | Linear 0→1.0 | Score ≥ 0.90 | Fix LR → Augment → Balance → Train×4 |

### MEDIUM ⭐⭐  
| Target | Grading | Success | Key Challenge |
|--------|---------|---------|---|
| Mean IoU ≥ 0.55 + No overfitting | Base×0.5 if overfit | Score ≥ 0.85 | **Prevent overfitting** while improving |

### HARD ⭐⭐⭐
| Component | Weight | Target | Penalty |
|-----------|--------|--------|---------|
| IoU Progress | 40% | ≥ 0.65 | Linear 0→0.4 |
| Sky Coverage | 30% | ≥ 0.50 | Linear 0→0.3 |
| Efficiency | 15% | ≥ 5 min left | 0.15 if met |
| Stability | 15% | No instability | 0.15 if met |

## Reward Components

```
Total = IoU Reward + Coverage Reward + Efficiency - Penalty

IoU Reward:
  +10 per 0.1 improvement
  -5 per 0.1 degradation

Coverage Reward:
  +2.0 for learning new class
  +5.0 per sky_iou improvement

Efficiency Reward:
  +0.5 if training gives good ROI
  -0.2 if training marginal returns
  +0.1 for config actions

Penalties:
  -1.0 if unstable
  -0.5 if overfitting
  -0.2 if redundant action
```

## Observation Report Fields

```
CURRENT METRICS:
  Mean IoU:        [0.0-1.0]  Overall performance
  Road IoU:        [0.0-1.0]  Critical for navigation
  Vegetation IoU:  [0.0-1.0]  2nd priority class
  Sky IoU:         [0.0-1.0]  Hardest, minority class
  Train Loss:      [value]    Decreasing = good
  Val Loss:        [value]    Should track train loss

TRAINING CONFIGURATION:
  Optimizer:       {SGD, ADAMW}
  Learning Rate:   [value]
  Batch Size:      {32, 16}
  L2 Regularization: [0, 0.0005]
  Data Augmentation: {Disabled, Enabled}
  Class Balancing:  {Disabled, Enabled}

ISSUES DETECTED:
  ⚠️  Training instability - suboptimal learning rate
  ⚠️  Overfitting detected - validation loss diverging
  ⚠️  Sky class not learning - severe class imbalance
  ✓   No major issues detected
```

## Common Patterns

### Solving Easy (Score 0.90+)
```
Step 1: adjust_learning_rate          [Fix instability]
Step 2: enable_augmentation           [Improve generalization]
Step 3: enable_class_balancing        [Learn sky]
Step 4: run_training_epoch            [Train]
Step 5: run_training_epoch
Step 6: run_training_epoch
Step 7: run_training_epoch
Result: Mean IoU ≈ 0.53 → Score 1.0
```

### Solving Medium (Score 0.85+)
```
Step 1: adjust_learning_rate          [Stability]
Step 2: enable_augmentation           [Generalization]
Step 3: increase_regularization       [Prevent overfitting]
Step 4: enable_class_balancing        [Sky class]
Step 5-8: run_training_epoch
Result: Mean IoU ≈ 0.57, No overfitting → Score 0.90
```

### Solving Hard (Score 0.80+)
```
Step 1: adjust_learning_rate          [CRITICAL]
Step 2: enable_class_balancing        [Sky 0% → ~20%]
Step 3: enable_augmentation           [+10% IoU]
Step 4: increase_regularization       [Stability]
Step 5-9: run_training_epoch          [5 epochs]
Result: Mean IoU ≈ 0.68, Sky ≈ 0.55, Stable → Score 0.82
```

## State Space

### Continuous Metrics (Observations)
- mean_iou: [0.0, 1.0]
- road_iou: [0.0, 1.0]
- vegetation_iou: [0.0, 1.0]
- sky_iou: [0.0, 1.0]
- epoch: [0, 50]
- compute_remaining_minutes: [0, 30]

### Boolean Flags
- overfitting: {true, false} - Val loss > train + margin
- instability: {true, false} - Bad learning rate
- early_stopped: {true, false} - Manual termination

### Hyperparameters
- optimizer: {sgd, adamw}
- batch_size: {32, 16}
- learning_rate: {0.01, 0.001}
- augmentation_level: {0, 2}
- regularization_strength: {0.0, 0.0005}
- class_balancing: {false, true}

## What Affects Each Class IoU

| Class | Baseline | Improves With | Hurt By | Bottleneck |
|-------|----------|---------------|---------|------------|
| Road | 0.40 | Training, Augment, LR fix | Instability | Learning plateau |
| Vegetation | 0.10 | Training, Augment | Nothing major | Minority class |
| Sky | 0.00 | **Class balancing** | Nothing (no gradients) | **Must enable balancing** |

## Episode Termination Conditions

| Condition | Result | Recovery |
|-----------|--------|----------|
| compute_remaining ≤ 0 | Game Over | Manage budget |
| early_stop_training | Game Over | Don't call |
| max_steps (50) reached | Game Over | Plan better |

## Debugging Checklist

### IoU Not Improving?
- [ ] Did you fix instability first? (adjust_learning_rate)
- [ ] Are you training? (run_training_epoch)
- [ ] Is it showing diminishing returns? (Already at plateau)

### Sky IoU Stuck at 0?
- [ ] **Enable class balancing!** (enable_class_balancing)
- Without it, sky never receives gradients

### Overfitting Detected?
- [ ] Increase regularization (increase_regularization)
- [ ] Enable augmentation (enable_augmentation)
- Both help with generalization

### Running Low on Compute?
- [ ] Stop training (early_stop_training)
- [ ] Or train fewer epochs
- Default: 30 min ÷ 5 min/epoch = 6 epochs max

### Training Unstable?
- [ ] Fix learning rate (adjust_learning_rate)
- Solves instability flag immediately

## Performance Expectations

### Random Agent (Baseline)
- Easy: 0.30-0.50 / 1.0
- Medium: 0.20-0.40 / 1.0
- Hard: 0.15-0.35 / 1.0

### Heuristic Agent (Greedy)
- Easy: 0.85-1.00 / 1.0
- Medium: 0.70-0.95 / 1.0
- Hard: 0.60-0.80 / 1.0

### Expert Agent (Optimal)
- Easy: 1.00 / 1.0
- Medium: 0.95+ / 1.0
- Hard: 0.85+ / 1.0

## Key Parameters (Hardcoded)

```python
# Improvement rates
base_improvement = 0.05        # Per epoch
aug_boost = 1.10               # +10% from augmentation
adamw_boost = 1.15             # +15% from AdamW
loss_decay_train = 0.92        # Per epoch
loss_decay_val = 0.93          # Per epoch

# Thresholds
overfitting_threshold = 0.3    # val_loss - train_loss
plateau_threshold = 0.01       # Min improvement to avoid plateau

# Costs
train_epoch_cost = 5           # Minutes base
batch_small_multiplier = 1.2   # Slower training with small batches

# Budget
initial_compute = 30           # Minutes
max_steps = 50                 # Per episode
```

## API Quick Start

```python
# Initialize
env = Environment(target_iou=0.55, initial_compute=30)

# Reset
obs = env.reset()

# Step
obs, reward, done, info = env.step(Action(action="run_training_epoch"))

# Grade
state = env.state()
score = HardTaskGrader.score(state)

# Properties
obs.mean_iou              # Current IoU
obs.report                # Full report
reward.value              # Total reward
done                      # Episode over?
```

## File Map

```
Core Environment:
  environment.py       → OpenEnv interface (reset/step/state)
  simulator.py         → ML dynamics engine
  models.py            → Pydantic schemas
  reward_engine.py     → Reward calculation

Grading:
  graders.py           → Task-specific graders
  easy_grader.py       → Easy task wrapper
  medium_grader.py     → Medium task wrapper
  hard_grader.py       → Hard task wrapper

Evaluation & Testing:
  evaluation.py        → Comprehensive evaluation
  test_environment.py  → 9 unit tests
  inference.py         → OpenAI agent interface

Documentation:
  ARCHITECTURE.md      → Design details
  USAGE.md             → Usage examples
  REFACTORING.md       → Changes summary
  This file: QUICK_REFERENCE.md
```

---

**For detailed information, refer to:**
- ARCHITECTURE.md (design philosophy)
- USAGE.md (practical examples)
- openenv.yaml (environment metadata)
