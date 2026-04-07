"""
Comprehensive evaluation module for OffroadSegNet ML environment.
Runs multiple seeds and evaluates performance across all task difficulties.
"""

import random
from typing import Dict, List, Tuple
from environment import Environment
from models import Action
from graders import EasyTaskGrader, MediumTaskGrader, HardTaskGrader


class EnvironmentEvaluator:
    """Evaluates agent performance and environment behavior."""
    
    def __init__(self, num_seeds: int = 3):
        self.num_seeds = num_seeds
        self.results = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
    
    def run_random_agent(self, task_name: str = "easy", num_episodes: int = 3) -> Dict[str, float]:
        """
        Run random agent on task for multiple episodes.
        Returns: success_rate, avg_score, avg_iou_improvement, avg_steps
        """
        action_names = [
            "adjust_learning_rate",
            "enable_augmentation",
            "increase_regularization",
            "switch_optimizer",
            "enable_class_balancing",
            "reduce_batch_size",
            "run_training_epoch",
            "early_stop_training",
        ]
        
        episode_scores = []
        episode_iou_improvements = []
        episode_steps = []
        
        for seed in range(self.num_seeds):
            random.seed(seed)
            env = Environment(seed=seed)
            obs = env.reset()
            
            total_iou_improvement = 0.0
            steps = 0
            
            for _ in range(40):  # Max steps per episode
                action = random.choice(action_names)
                obs, reward, done, info = env.step(Action(action=action))
                total_iou_improvement += info.get("iou_improvement", 0.0)
                steps += 1
                
                if done:
                    break
            
            # Score based on task difficulty
            state = env.state()
            if task_name == "easy":
                score = EasyTaskGrader.score(state)
            elif task_name == "medium":
                score = MediumTaskGrader.score(state)
            else:  # hard
                score = HardTaskGrader.score(state)
            
            episode_scores.append(score)
            episode_iou_improvements.append(total_iou_improvement)
            episode_steps.append(steps)
        
        success_count = sum(1 for s in episode_scores if s >= 0.9)
        
        return {
            "success_rate": success_count / len(episode_scores),
            "avg_score": sum(episode_scores) / len(episode_scores),
            "avg_iou_improvement": sum(episode_iou_improvements) / len(episode_iou_improvements),
            "avg_steps": sum(episode_steps) / len(episode_steps),
            "scores": episode_scores,
        }
    
    def evaluate_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """Evaluate performance across all task difficulties."""
        print("\n" + "=" * 70)
        print("OFFROAD SEGMENTATION ENVIRONMENT EVALUATION")
        print("=" * 70)
        
        results = {}
        
        for task_name in ["easy", "medium", "hard"]:
            print(f"\n📊 Evaluating {task_name.upper()} Task (Random Agent)...")
            metrics = self.run_random_agent(task_name, num_episodes=self.num_seeds)
            results[task_name] = metrics
            
            print(f"  Success Rate:          {metrics['success_rate']:.1%}")
            print(f"  Average Task Score:    {metrics['avg_score']:.4f}/1.0")
            print(f"  Avg IoU Improvement:   {metrics['avg_iou_improvement']:.4f}")
            print(f"  Avg Steps to Complete: {metrics['avg_steps']:.1f}")
            print(f"  Seed Scores:           {[f'{s:.3f}' for s in metrics['scores']]}")
        
        # Overall statistics
        all_scores = [s for task in results.values() for s in task["scores"]]
        avg_overall = sum(all_scores) / len(all_scores) if all_scores else 0
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Overall Average Score:   {avg_overall:.4f}/1.0")
        print(f"Easy Task:               {results['easy']['avg_score']:.4f}/1.0")
        print(f"Medium Task:             {results['medium']['avg_score']:.4f}/1.0")
        print(f"Hard Task:               {results['hard']['avg_score']:.4f}/1.0")
        print(f"Total Episodes Run:      {len(all_scores)}")
        print("=" * 70 + "\n")
        
        return results
    
    def test_determinism(self) -> bool:
        """
        Verify that environment is deterministic with same seed.
        """
        print("\n🔬Testing Determinism...")
        
        seed = 42
        
        # Run 1
        random.seed(seed)
        env1 = Environment(seed=seed)
        obs1 = env1.reset()
        trajectory1 = [(obs1.mean_iou, obs1.overfitting, obs1.instability)]
        
        for _ in range(5):
            obs1, _, _, _ = env1.step(Action(action="run_training_epoch"))
            trajectory1.append((obs1.mean_iou, obs1.overfitting, obs1.instability))
        
        # Run 2
        random.seed(seed)
        env2 = Environment(seed=seed)
        obs2 = env2.reset()
        trajectory2 = [(obs2.mean_iou, obs2.overfitting, obs2.instability)]
        
        for _ in range(5):
            obs2, _, _, _ = env2.step(Action(action="run_training_epoch"))
            trajectory2.append((obs2.mean_iou, obs2.overfitting, obs2.instability))
        
        # Compare
        deterministic = trajectory1 == trajectory2
        
        if deterministic:
            print("  ✓ Environment is deterministic")
        else:
            print("  ✗ Environment is NOT deterministic")
            print(f"    Trajectory 1: {trajectory1[:2]}")
            print(f"    Trajectory 2: {trajectory2[:2]}")
        
        return deterministic


def main():
    """Main evaluation entry point."""
    evaluator = EnvironmentEvaluator(num_seeds=3)
    
    # Check determinism
    evaluator.test_determinism()
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_all_tasks()
    
    # Print interpretation
    print("\n📝 INTERPRETATION:")
    print("-" * 70)
    print("""
Each task represents increasing difficulty in ML engineering:

EASY:  Achieve baseline IoU improvement (0.50 target)
       - Requires: Basic training and configuration

MEDIUM: Achieve better IoU while preventing overfitting (0.55 target)
        - Requires: Configuration + stable training + regularization

HARD:  Multi-objective: high IoU (0.65) + class coverage (sky 0.50)
       + compute efficiency + stability
       - Requires: Expert configuration sequencing and resource management

Random agent performance indicates environment balance.
Scores correlate with agent quality (heuristic > random > naive).
    """)


if __name__ == "__main__":
    main()
