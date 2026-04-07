# Offroad Segmentation Environment

This project has been refactored into a deterministic, lightweight OpenEnv simulation environment. The goal is to provide a realistic ML engineering debugging task without the overhead of heavy computational models or large datasets.

## Goal

Act as an ML systems engineer to improve the `OffroadSegNet`'s performance. The model initially struggles with instability, missing class coverage (Sky), and general poor validation IoU limit.

The environment simulates training progress and metrics sequentially.

## Setup

Ensure you have the base requirements:
```bash
pip install pydantic openai
```

## Environment Architecture

- `models.py`: Pydantic BaseModels for `Observation`, `Action`, and `Reward` defining the communication interface.
- `simulator.py`: Purely deterministic state-engine simulating hyperparameters, model capacity scaling, learning rate instability, and domain generalization metrics.
- `environment.py`: Contains the OpenEnv compliant `reset()`, `step()`, and `state()` endpoints wrapper class.
- `inference.py`: Baseline inference script using an OpenAI agent to interact with the environment contextually.

### Actions

You have a total of 8 actions mimicking sequential ML tracking:
- `adjust_learning_rate`
- `enable_augmentation`
- `increase_regularization`
- `switch_optimizer`
- `enable_class_balancing`
- `reduce_batch_size`
- `run_training_epoch`
- `early_stop_training`

### Tasks

Three integrated tasks are provided in `openenv.yaml` to test agent competence:
1. **Easy:** Reach basic acceptable Mean IoU performance (0.50). Evaluated by `easy_grader.py`.
2. **Medium:** Balance high IoU against simulated overfitting thresholds. Evaluated by `medium_grader.py`.
3. **Hard:** Attain high IoU mapping specifically catching the missing zero-coverage categories (e.g. Sky domain) while maintaining compute efficiency. Evaluated by `hard_grader.py`.

## Running the Agent baseline

Provide your environment variables and execute the baseline:

```powershell
$env:OPENAI_API_KEY="your_key"
$env:MODEL_NAME="gpt-4o"
python inference.py
```

## Testing

A suite of stability and property tests verifies continuous validity of the simulator.

```bash
pytest test_environment.py
```
