# OffroadSegNet ML Debugging Environment - Architecture Guide

## Overview

This refactored project transforms a generic hyperparameter tuning simulator into a **realistic ML engineering workflow environment**. The agent acts as an ML engineer debugging an OffroadSegNet semantic segmentation model, solving real optimization problems encountered in production ML systems.

## Architecture Changes

### 1. Simulator Realism (`simulator.py`)

**Before:** Abstract toggle-based state with binary flags and linear improvements

**After:** Realistic ML dynamics with:
- **ML Pipeline Parameters**: backbone_depth, feature_channels, regularization_strength, learning_rate, optimizer, batch_size, augmentation_level, mixed_precision, class_balancing, loss_weights
- **Realistic Training Curves**: 
  - Diminishing returns as training progresses
  - Training plateau after convergence
  - Overfitting curves based on loss divergence
  - Instability from suboptimal learning rates
- **Class Imbalance Effects**: Sky class doesn't learn without balancing
- **Augmentation Benefits**: Improves generalization by 10-15%
- **Deterministic Behavior**: Seeded RNG for reproducibility

**Key Simulation Features:**
```
Loss dynamics:     train_loss * 0.92, val_loss * 0.93 per epoch
Learning curves:   steep → plateau (diminishing returns)
Overfitting risk:  detected when val_loss - train_loss > 0.3
Instability:       causes -5% to -8% IoU regression per epoch
Class learning:    sky_iou = 0.0 without balancing, improves with focal loss
```

### 2. Realistic Observations (`models.py` + `simulator.py`)

**Before:** Basic one-liner with 5 metrics

**After:** Structured experiment report showing:
```
OFFROAD SEGMENTATION EXPERIMENT REPORT
=====================================
Model: OffroadSegNet ResNet-18
Dataset: Offroad Terrain Segmentation (Train + Val)
Epoch: 5 | Compute Budget: 20m remaining

CURRENT METRICS:
  Mean IoU:        0.2847
  Road IoU:        0.5432
  Vegetation IoU:  0.2156
  Sky IoU:         0.0000 (missing coverage)
  Train Loss:      0.9234
  Val Loss:        1.0156

TRAINING CONFIGURATION:
  Optimizer:       ADAMW
  Learning Rate:   0.001
  Class Balancing: Disabled
  L2 Regularization: 0.0005
  ...

ISSUES DETECTED:
  ⚠️ Sky class not learning - severe class imbalance
  ✓ Instability resolved
```

### 3. Advanced Grading System

**Created:** `graders.py` with class-based graders

#### EasyTaskGrader
- **Target:** Mean IoU ≥ 0.50
- **Scoring:** Linear 0 → 1.0 as IoU improves
- **Focus:** Basic debugging and training

#### MediumTaskGrader  
- **Target:** Mean IoU ≥ 0.55 + No overfitting
- **Scoring:** Base score * 0.5 if overfitting detected
- **Focus:** Generalization and stability
- **Challenge:** Balance performance with preventing overfitting

#### HardTaskGrader
- **Multi-objective optimization:**
  - IoU progress (40%): Target 0.65
  - Sky class coverage (30%): Target 0.50 (challenging minority)
  - Efficiency (15%): Need ≥5 min compute remaining
  - Stability (15%): No instability during training
- **Scoring:** Weighted sum of components
- **Challenge:** Expert-level simultaneous optimization

### 4. Intelligent Reward System (`reward_engine.py`)

**Created:** `RewardEngine` class for nuanced feedback

**Reward Components:**
- **IoU Reward**: +10 per 0.1 improvement, -5 per 0.1 degradation
- **Coverage Reward**: +2.0 for learning new classes, +5 per sky_iou improvement
- **Efficiency Reward**: +0.5 for good cost-benefit training, -0.2 for poor returns
- **Penalties**: -1.0 for instability, -0.5 for overfitting, -0.2 for repeated actions

**Result:** Smooth trajectory signal guiding agent toward optimal decisions

### 5. ML-Realistic Actions

**Before:** Generic hyperparameter adjustments

**After:** Real ML engineering decisions:
- `adjust_learning_rate`: Fix 0.01 → 0.001 (critical for stability)
- `enable_augmentation`: Apply rotation, flip, color jitter (10% IoU boost)
- `increase_regularization`: Add L2 reg to prevent overfitting
- `switch_optimizer`: SGD → AdamW (15% faster convergence)
- `enable_class_balancing`: Focal loss + reweighting (enables sky class learning)
- `reduce_batch_size`: 32 → 16 (memory efficiency, slower convergence)
- `run_training_epoch`: Actual training step (costs 5-6 min compute)
- `early_stop_training`: Preserve compute budget

### 6. Comprehensive Testing (`test_environment.py`)

**Enhanced test suite covering:**
- ✓ Crash-free operation
- ✓ Random agent behavior
- ✓ Heuristic agent performance
- ✓ Reward monotonicity
- ✓ Environment stability (100+ steps)
- ✓ Long episode handling
- ✓ Grader consistency
- ✓ Action validation
- ✓ Observation formatting

### 7. Sophisticated Evaluation (`evaluation.py`)

**New:** Comprehensive evaluation module with:
- Random agent baseline across all tasks
- Multi-seed evaluation (3 seeds for statistical significance)
- Determinism verification
- Success rate tracking
- Performance interpretation and failure mode analysis

### 8. OpenAI Integration (`inference.py`)

**Refactored:** Modern inference pipeline with:
- OpenAI API integration for agent decision-making
- JSON response parsing for action selection
- Heuristic fallback when API unavailable
- Structural logs showing:
  - Per-step actions and rewards
  - Task-by-task evaluation
  - Score interpretation guide

### 9. OpenEnv Compliance (`openenv.yaml`)

**Updated:** Comprehensive metadata with:
- Task descriptions with difficulty levels (⭐ to ⭐⭐⭐)
- Success thresholds and grading logic
- All supported actions documented
- Observation features explained
- Reward components detailed
- Expected performance ranges

## Key Improvements Over Original

| Aspect | Before | After |
|--------|--------|-------|
| **Observations** | Basic one-liner | Structured 500+ char reports |
| **Grading** | Function-based | Class-based with logic |
| **Realism** | Abstract toggles | Actual ML parameters |
| **Dynamics** | Linear improvements | Realistic learning curves |
| **Rewards** | Simple scalar | Multi-component with penalties |
| **Testing** | 4 basic tests | 9 comprehensive tests |
| **Evaluation** | None | Multi-task evaluation |
| **Documentation** | Minimal | Extensive |
| **OpenEnv** | Basic | Full v2.0 spec |

## Task Difficulty Progression

### Easy (⭐)
- **Problem:** Baseline model performs poorly
- **Challenge:** Configure parameters and train to reach 0.50 IoU
- **Optimal strategy:** 3-4 fixes + 4 epochs
- **Expected score:** 0.90+

### Medium (⭐⭐)
- **Problem:** Can improve but easily overfit
- **Challenge:** Reach 0.55 IoU without overfitting
- **Optimal strategy:** Enable augmentation + regularization, monitor val loss
- **Expected score:** 0.85+

### Hard (⭐⭐⭐)
- **Problem:** Multi-objective with conflicting goals
- **Challenge:** Simultaneous optimization:
  1. High mean IoU (0.65)
  2. Learn missing sky class (0.50)
  3. Use compute efficiently
  4. Maintain stability
- **Optimal strategy:** Fix instability → enable balancing → careful training + efficiency
- **Expected score:** 0.80+

## Default Baseline

**Initial State:**
- Road IoU: 0.40 (somewhat learned)
- Vegetation IoU: 0.10 (underlearned)
- Sky IoU: 0.00 (missing completely)
- Mean IoU: 0.167
- Learning rate: 0.01 (causes instability)
- Class balancing: Disabled (sky never learns)
- Overfitting: Not yet, but risk with training

**Common Issues:**
1. **Instability** - Bad learning rate causes loss spikes
2. **Class imbalance** - Sky never learns without balancing
3. **Overfitting** - Too many epochs without regularization
4. **Inefficiency** - Wasted training epochs with minimal gains

## File Structure

```
.
├── environment.py          # OpenEnv-compliant environment
├── simulator.py            # Realistic ML dynamics engine
├── models.py               # Pydantic schemas (Action, Observation, Reward)
├── reward_engine.py        # Intelligent multi-component rewards
├── graders.py              # Task-specific graders (Easy/Medium/Hard)
├── inference.py            # OpenAI agent interface
├── evaluation.py           # Comprehensive evaluation suite
├── test_environment.py     # 9 comprehensive tests
├── easy_grader.py          # Easy task wrapper
├── medium_grader.py        # Medium task wrapper
├── hard_grader.py          # Hard task wrapper
└── openenv.yaml            # OpenEnv v2.0 metadata
```

## Integration Points

### With External Agents
```python
from environment import Environment
from models import Action

env = Environment(target_iou=0.50)
obs = env.reset()

for _ in range(max_steps):
    # Agent makes decision based on obs.report
    action = agent.decide(obs)
    obs, reward, done, info = env.step(Action(action=action))
    if done:
        break

state = env.state()
score = grade_hard(state)  # Task grading
```

### With OpenAI API
```python
os.environ["OPENAI_API_KEY"] = "sk-..."
python inference.py
# Runs agent through all 3 tasks with detailed logging
```

### Evaluation
```python
from evaluation import EnvironmentEvaluator

evaluator = EnvironmentEvaluator(num_seeds=10)
results = evaluator.evaluate_all_tasks()
```

## Performance Characteristics

**Determinism:** ✓ Fully deterministic with seed control
**Efficiency:** ✓ Single-step execution < 1ms
**Scalability:** ✓ Supports 1000+ episode sequences
**Stability:** ✓ No crashes in 100+ step episodes

## Backward Compatibility

- Original grader functions remain supported
- Action/Observation/Reward schemas unchanged
- Environment API (reset/step/state) preserved
- OpenEnv entry points maintained

## Future Extensions

Possible enhancements while maintaining architecture:
- Curriculum learning (progressive difficulty)
- Multi-model analysis (compare architectures)
- Budget uncertainty (variable compute costs)
- Dynamic task generation
- Auxiliary loss terms
- Data quality variations
