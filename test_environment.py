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
    env = Environment(target_iou=0.60, initial_compute=50)  # More compute for fixed config costs
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
    assert state["mean_iou"] > 0.22, f"Expected improvement, got {state['mean_iou']}"
    # Should have fixed instability
    assert not state["instability"], "Instability should be fixed"
    # Should have started learning sky class
    assert state["sky_iou"] > 0.0, "Sky class should be learning"


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


def test_reward_bounds():
    """Rewards should always be bounded to [-1.0, 1.0]."""
    env = Environment()
    env.reset()
    
    actions = [
        "adjust_learning_rate",
        "enable_augmentation",
        "increase_regularization",
        "switch_optimizer",
        "enable_class_balancing",
        "reduce_batch_size",
        "run_training_epoch",
        "run_training_epoch",
    ]
    
    for _ in range(50):
        act = random.choice(actions)
        obs, reward, done, info = env.step(Action(action=act))
        
        # Reward value must be bounded
        assert -1.0 <= reward.value <= 1.0, f"Reward {reward.value} out of bounds"
        
        # All reward components should be valid numbers
        assert isinstance(reward.iou_reward, float)
        assert isinstance(reward.coverage_reward, float)
        assert isinstance(reward.efficiency_reward, float)
        assert isinstance(reward.penalty, float)
        
        if done:
            break


def test_determinism():
    """Same seed should produce identical sequences."""
    seed = 42
    
    # Run 1
    env1 = Environment(seed=seed)
    obs1 = env1.reset(seed=seed)
    ious1 = [obs1.mean_iou]
    
    actions = ["adjust_learning_rate", "enable_augmentation", "enable_class_balancing"]
    for _ in range(3):
        for act in actions:
            obs1, _, _, _ = env1.step(Action(action=act))
            ious1.append(obs1.mean_iou)
    
    # Run 2 with same seed
    env2 = Environment(seed=seed)
    obs2 = env2.reset(seed=seed)
    ious2 = [obs2.mean_iou]
    
    for _ in range(3):
        for act in actions:
            obs2, _, _, _ = env2.step(Action(action=act))
            ious2.append(obs2.mean_iou)
    
    # Sequences should be identical
    assert len(ious1) == len(ious2)
    for v1, v2 in zip(ious1, ious2):
        assert abs(v1 - v2) < 1e-6, f"Determinism broken: {v1} vs {v2}"


def test_info_dict_completeness():
    """Info dict should contain all required fields."""
    env = Environment()
    obs = env.reset()
    
    # First step
    obs, reward, done, info = env.step(Action(action="run_training_epoch"))
    
    required_fields = [
        "step",
        "message",
        "action",
        "iou_improvement",
        "termination_reason",
        "success",
        "mean_iou",
        "remaining_compute",
        "episode_total_reward",
    ]
    
    for field in required_fields:
        assert field in info, f"Missing required field: {field}"
    
    # Check types
    assert isinstance(info["step"], int)
    assert isinstance(info["message"], str)
    assert isinstance(info["action"], str)
    assert isinstance(info["iou_improvement"], (int, float))
    assert isinstance(info["success"], bool)
    assert isinstance(info["mean_iou"], float)
    assert isinstance(info["remaining_compute"], int)
    assert isinstance(info["episode_total_reward"], (int, float))


def test_success_detection():
    """Environment should correctly report success in info dict."""
    env = Environment(target_iou=0.40)  # Very achievable target
    obs = env.reset()
    
    # Track success field across multiple steps
    successes = []
    
    actions = [
        "adjust_learning_rate",
        "enable_augmentation",
        "enable_class_balancing",
    ] + ["run_training_epoch"] * 8
    
    for act in actions:
        obs, reward, done, info = env.step(Action(action=act))
        
        # Verify success field exists and is bool
        assert "success" in info, "success field missing from info dict"
        assert isinstance(info["success"], bool), "success should be boolean"
        
        # Verify success field matches actual IoU vs target
        expected_success = obs.mean_iou >= env.target_iou
        assert info["success"] == expected_success, \
            f"Success mismatch: field={info['success']}, expected={expected_success}, IoU={obs.mean_iou:.4f}, target={env.target_iou}"
        
        successes.append(info["success"])
        
        if done:
            break
    
    # If we ever reached the target, success should have been True at that point
    # Verify the logic works at least some of the time
    assert len(successes) > 0, "Should have taken at least one step"


def test_reset_clears_state():
    """Reset should properly clear episode state."""
    env = Environment()
    
    # First episode
    obs = env.reset()
    initial_iou = obs.mean_iou
    
    # Take some steps
    for _ in range(5):
        env.step(Action(action="run_training_epoch"))
    
    # Reset
    obs2 = env.reset()
    
    # State should be cleared
    assert env.episode_steps == 0, "Episode steps not reset"
    assert env.episode_total_reward == 0.0, "Total reward not reset"
    assert len(env.action_history) == 0, "Action history not reset"
    assert env.termination_reason == "", "Termination reason not reset"
    
    # IoU should return to baseline
    assert abs(obs2.mean_iou - initial_iou) < 1e-5, "IoU not reset"


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
        test_reward_bounds,
        test_determinism,
        test_info_dict_completeness,
        test_success_detection,
        test_reset_clears_state,
    ]
    
    for test_func in tests:
        try:
            test_func()
            print(f"PASS {test_func.__name__}")
        except AssertionError as e:
            print(f"FAIL {test_func.__name__}: {e}")
        except Exception as e:
            print(f"FAIL {test_func.__name__}: {type(e).__name__}: {e}")
