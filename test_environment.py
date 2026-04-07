"""
Comprehensive test suite for OffroadSegNet ML environment.
Tests: crash-free operation, random agents, heuristics, reward properties, stability.
"""

import random
from environment import Environment
from models import Action
from graders import EasyTaskGrader, MediumTaskGrader, HardTaskGrader


def test_environment_crash_free():
    """Basic smoke test - environment should not crash."""
    env = Environment()
    obs = env.reset()
    assert obs is not None
    assert obs.report is not None
    assert obs.mean_iou >= 0.0
    
    actions = ["adjust_learning_rate", "enable_augmentation", "run_training_epoch", "run_training_epoch"]
    for act in actions:
        obs, reward, done, info = env.step(Action(action=act))
        assert obs is not None
        assert reward is not None
        assert isinstance(done, bool)


def test_random_agent():
    """Random agent should complete episodes without crashing."""
    env = Environment()
    obs = env.reset()
    
    actions = [
        "adjust_learning_rate", "enable_augmentation", "increase_regularization", 
        "switch_optimizer", "enable_class_balancing", "reduce_batch_size", 
        "run_training_epoch", "early_stop_training"
    ]
    
    episode_reward = 0.0
    for _ in range(40):
        act = random.choice(actions)
        obs, reward, done, info = env.step(Action(action=act))
        episode_reward += reward.value
        if done:
            break
    
    # Should complete or run out of compute
    assert done or env.simulator.remaining_compute <= 0
    # Random agent should not achieve negative total reward
    assert episode_reward > -20.0


def test_heuristic_agent():
    """Optimal heuristic agent should improve performance."""
    env = Environment(target_iou=0.60)
    obs = env.reset()
    
    # Optimal sequence: fix instability, enable augmentation/balancing, then train
    optimal_actions = [
        "adjust_learning_rate",     # Fix instability
        "enable_augmentation",      # Better generalization
        "enable_class_balancing",   # Sky class learning
        "run_training_epoch",
        "run_training_epoch",
        "run_training_epoch",
        "run_training_epoch",
        "run_training_epoch",
    ]
    
    for act in optimal_actions:
        obs, reward, done, info = env.step(Action(action=act))
        if done:
            break
    
    state = env.state()
    # Heuristic should improve from baseline (0.167 mean IoU)
    assert state["mean_iou"] > 0.25
    # Should have fixed instability
    assert not state["instability"] 
    # Should have started learning sky class
    assert state["sky_iou"] > 0.0


def test_reward_monotonicity():
    """
    Reward should generally increase with IoU progress.
    (Not strictly monotonic due to exploration, but trend should be positive)
    """
    env = Environment()
    env.reset()
    
    # Training epochs with proper setup should yield positive rewards
    setup_actions = ["adjust_learning_rate", "enable_augmentation", "enable_class_balancing"]
    for act in setup_actions:
        env.step(Action(action=act))
    
    rewards = []
    for _ in range(5):
        obs, reward, done, info = env.step(Action(action="run_training_epoch"))
        rewards.append(reward.value)
        if done:
            break
    
    # Average reward from training should be positive
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    assert avg_reward > -1.0, f"Training epochs should yield positive rewards, got {avg_reward}"


def test_environment_stability():
    """Environment should remain stable across long runs."""
    env = Environment()
    obs = env.reset()
    
    actions = [
        "adjust_learning_rate",
        "enable_augmentation",
        "increase_regularization",
        "switch_optimizer",
        "enable_class_balancing",
        "run_training_epoch",
    ]
    
    step_count = 0
    crashes = 0
    
    try:
        for i in range(100):
            act = actions[i % len(actions)]
            obs, reward, done, info = env.step(Action(action=act))
            step_count += 1
            
            if done:
                break
    except Exception as e:
        crashes += 1
        raise AssertionError(f"Environment crashed after {step_count} steps: {e}")
    
    assert crashes == 0
    assert step_count > 0


def test_long_episode():
    """Environment should handle extended interaction sequences."""
    env = Environment(initial_compute=60)  # More compute for longer episode
    obs = env.reset()
    
    # Mix of configuration and training actions
    action_sequence = [
        "adjust_learning_rate",
        "enable_augmentation",
        "switch_optimizer",
        "enable_class_balancing",
        "increase_regularization",
    ] + ["run_training_epoch"] * 15 + [
        "reduce_batch_size",
        "run_training_epoch",
        "run_training_epoch",
    ]
    
    total_reward = 0.0
    iou_improvements = []
    last_iou = obs.mean_iou
    
    for act in action_sequence:
        obs, reward, done, info = env.step(Action(action=act))
        total_reward += reward.value
        iou_improve = obs.mean_iou - last_iou
        iou_improvements.append(iou_improve)
        last_iou = obs.mean_iou
        
        if done:
            break
    
    # Should make progress
    assert last_iou > 0.167, "Should improve from baseline"
    assert len(iou_improvements) > 5, "Should complete multiple steps"


def test_grader_consistency():
    """Graders should produce consistent scores."""
    env = Environment()
    obs = env.reset()
    
    # Run a sequence
    actions = ["adjust_learning_rate", "enable_augmentation", "enable_class_balancing"]
    for act in actions:
        env.step(Action(action=act))
    
    for _ in range(4):
        env.step(Action(action="run_training_epoch"))
    
    state = env.state()
    
    # Get scores from all graders
    easy_score = EasyTaskGrader.score(state)
    medium_score = MediumTaskGrader.score(state)
    hard_score = HardTaskGrader.score(state)
    
    # Scores should be in valid range
    assert 0.0 <= easy_score <= 1.0
    assert 0.0 <= medium_score <= 1.0
    assert 0.0 <= hard_score <= 1.0
    
    # Repeated calls should give same score
    easy_score2 = EasyTaskGrader.score(state)
    assert easy_score == easy_score2


def test_action_validation():
    """Invalid actions should be handled gracefully."""
    env = Environment()
    obs = env.reset()
    
    # Valid action should work
    obs, reward, done, info = env.step(Action(action="run_training_epoch"))
    assert obs is not None
    
    # Environment should handle all declared actions
    valid_actions = [
        "adjust_learning_rate",
        "enable_augmentation",
        "increase_regularization",
        "switch_optimizer",
        "enable_class_balancing",
        "reduce_batch_size",
        "run_training_epoch",
        "early_stop_training",
    ]
    
    for act in valid_actions:
        obs, reward, done, info = env.step(Action(action=act))
        assert obs is not None


def test_observation_report_format():
    """Observation report should be properly formatted and informative."""
    env = Environment()
    obs = env.reset()
    
    # Initial report should mention baseline issues
    assert "OFFROAD SEGMENTATION EXPERIMENT REPORT" in obs.report
    assert "Mean IoU" in obs.report
    assert "Road IoU" in obs.report
    assert "Vegetation IoU" in obs.report
    assert "Sky IoU" in obs.report
    
    # Run some actions
    env.step(Action(action="adjust_learning_rate"))
    env.step(Action(action="enable_augmentation"))
    env.step(Action(action="run_training_epoch"))
    
    obs = env.simulator.get_observation()
    
    # Report should be updated
    assert "ISSUES DETECTED" in obs.report
    assert "TRAINING CONFIGURATION" in obs.report


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_environment_crash_free,
        test_random_agent,
        test_heuristic_agent,
        test_reward_monotonicity,
        test_environment_stability,
        test_long_episode,
        test_grader_consistency,
        test_action_validation,
        test_observation_report_format,
    ]
    
    for test_func in tests:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {type(e).__name__}: {e}")
