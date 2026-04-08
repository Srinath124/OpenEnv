"""
ML inference engine using OpenAI API for agent decision-making.
Evaluates agent performance on OffroadSegNet ML environment across all task difficulties.
Produces structured logs and baseline scores.
"""

import os
import json
from typing import Optional
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from environment import Environment
from models import Action
from graders import EasyTaskGrader, MediumTaskGrader, HardTaskGrader


class MLAgentInference:
    """Uses OpenAI API to drive ML engineering decisions."""
    
    def __init__(self, model: str = None, max_tokens: int = 100):
        # Read environment variables safely
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model or os.environ.get("MODEL_NAME", "gpt-4o-mini")
        self.max_tokens = max_tokens
        
        # Initialize client only if API key available
        self.client = None
        if self.api_key and OpenAI:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except TypeError as e:
                print(f"[WARNING] Failed to initialize OpenAI client - invalid API key: {e}")
            except Exception as e:
                print(f"[WARNING] Failed to initialize OpenAI client: {e}")
    
    def get_action(self, observation_report: str) -> Optional[str]:
        """
        Query OpenAI API for next action given experiment report.
        Returns action name or None if API unavailable.
        """
        if not self.client:
            return None
        
        try:
            if not observation_report or not isinstance(observation_report, str):
                return None
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert ML engineer debugging an OffroadSegNet semantic segmentation model. "
                            "Analyze the experiment report and recommend the next action to improve IoU. "
                            "Output ONLY valid JSON with a single 'action' field. "
                            "Valid actions: adjust_learning_rate, enable_augmentation, increase_regularization, "
                            "switch_optimizer, enable_class_balancing, reduce_batch_size, run_training_epoch, early_stop_training"
                        )
                    },
                    {"role": "user", "content": observation_report}
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                temperature=0.2,
            )
            
            content = response.choices[0].message.content
            action_dict = json.loads(content)
            return action_dict.get("action")
        
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON decode error: {e}")
            return None
        except AttributeError as e:
            print(f"[WARNING] Attribute error in API response: {e}")
            return None
        except Exception as e:
            print(f"[WARNING] API Error: {e}")
            return None
    
    def heuristic_action(self, observation_report: str) -> str:
        """
        Fallback heuristic when API is unavailable.
        Simple rule-based strategy for debugging.
        Always returns a valid action.
        """
        import re
        
        try:
            if not observation_report or not isinstance(observation_report, str):
                return "run_training_epoch"
            
            report_lower = observation_report.lower()
            
            # Extract mean IoU from report
            mean_iou = 0.0
            if "mean iou" in report_lower or "mean_iou" in report_lower:
                try:
                    match = re.search(r"mean[_\s]+iou[:\s]+([0-9.]+)", report_lower)
                    if match:
                        mean_iou = float(match.group(1))
                except (AttributeError, ValueError, TypeError):
                    mean_iou = 0.0
            
            # Early stop if target reached (generic 0.50 threshold)
            if mean_iou >= 0.50:
                return "early_stop_training"
            
            # Sky class missing (0.0000) - enable balancing first priority
            if ("sky_iou" in report_lower or "sky" in report_lower) and "0.0000" in observation_report:
                return "enable_class_balancing"
            
            # Instability is critical - fix it first
            if "instability" in report_lower or "suboptimal" in report_lower:
                return "adjust_learning_rate"
            
            # Overfitting - add regularization
            if "overfitting" in report_lower or "diverging" in report_lower:
                return "increase_regularization"
            
            # Plateau or poor generalization - enable augmentation
            if "plateau" in report_lower or "generalization" in report_lower:
                return "enable_augmentation"
            
            # Default safe action (always valid)
            return "run_training_epoch"
        
        except Exception as e:
            # Log error and return safe default
            print(f"[WARNING] Heuristic action error: {e}")
            return "run_training_epoch"


def run_task_evaluation(task_name: str = "easy", max_steps: int = 15, verbose: bool = True) -> dict:
    """
    Run complete task evaluation with OpenAI agent or heuristic fallback.
    
    Args:
        task_name: "easy", "medium", or "hard"
        max_steps: Maximum steps per episode
        verbose: Print progress
        
    Returns:
        dict with: final_state, task_score, trajectory, steps_taken, total_reward
    """
    try:
        if verbose:
            print(f"\n{'='*70}")
            print(f"[START] Task: {task_name.upper()}")
            print(f"{'='*70}")
        
        # Task-specific configuration
        task_config = {
            "easy": {"target_iou": 0.50, "grader": EasyTaskGrader},
            "medium": {"target_iou": 0.55, "grader": MediumTaskGrader},
            "hard": {"target_iou": 0.65, "grader": HardTaskGrader}
        }
        
        if task_name not in task_config:
            raise ValueError(f"Unknown task: {task_name}")
        
        config = task_config[task_name]
        
        # Initialize environment
        env = Environment(
            target_iou=config["target_iou"],
            initial_compute=30,
            seed=42,
            debug=False,
            max_steps=max_steps
        )
        
        obs = env.reset()
        
        # Initialize agent
        agent = MLAgentInference()
        
        # Trajectory tracking
        trajectory = []
        episode_reward = 0.0
        
        # Run episode
        for step in range(max_steps):
            try:
                # Get action from agent
                action_name = agent.get_action(obs.report) if agent.client else agent.heuristic_action(obs.report)
                if action_name is None:
                    action_name = agent.heuristic_action(obs.report)
                
                # Execute action
                try:
                    obs, reward, done, info = env.step(Action(action=action_name))
                except (ValueError, AttributeError, TypeError):
                    # Fallback on invalid action
                    action_name = "run_training_epoch"
                    obs, reward, done, info = env.step(Action(action=action_name))
                
                episode_reward += reward.value
                
                # Log step
                step_log = {
                    "step": step + 1,
                    "action": action_name,
                    "reward": reward.value,
                    "mean_iou": obs.mean_iou,
                    "compute_remaining": obs.compute_remaining_minutes,
                }
                trajectory.append(step_log)
                
                if verbose and ((step + 1) % 5 == 0 or done):
                    print(f"[STEP {step+1}] {action_name}: "
                          f"IoU={obs.mean_iou:.4f}, Reward={reward.value:+.3f}, "
                          f"Compute={obs.compute_remaining_minutes}min")
                
                if done:
                    break
            
            except Exception as e:
                print(f"[WARNING] Error at step {step+1}: {e}")
                continue
        
        # Get final state and evaluation
        final_state = env.state()
        final_score = config["grader"].score(final_state)
        
        result = {
            "task": task_name,
            "final_state": final_state,
            "task_score": final_score,
            "trajectory": trajectory,
            "steps_taken": len(trajectory),
            "total_reward": episode_reward,
            "final_iou": obs.mean_iou,
            "termination_reason": env.termination_reason
        }
        
        if verbose:
            print(f"[END] Final IoU: {obs.mean_iou:.4f}, Task Score: {final_score:.3f}, "
                  f"Steps: {len(trajectory)}, Total Reward: {episode_reward:.2f}")
            print(f"[END] Termination: {env.termination_reason}")
            print(f"{'='*70}\n")
        
        return result
    
    except Exception as e:
        print(f"[ERROR] Task evaluation failed: {e}")
        return {
            "task": task_name,
            "final_state": None,
            "task_score": 0.0,
            "trajectory": [],
            "steps_taken": 0,
            "total_reward": 0.0,
            "final_iou": 0.0,
            "termination_reason": f"Error: {str(e)}"
        }


def evaluate_all_tasks(max_steps: int = 15, verbose: bool = True) -> dict:
    """
    Run evaluation on all task difficulties.
    Returns summary statistics across all tasks.
    """
    print(f"\n[EVALUATION] Starting comprehensive task evaluation")
    print(f"[EVALUATION] Timestamp: {datetime.now().isoformat()}")
    print(f"[EVALUATION] Max steps per task: {max_steps}")
    
    results = {}
    task_targets = {"easy": 0.50, "medium": 0.55, "hard": 0.65}
    
    for task_name in ["easy", "medium", "hard"]:
        results[task_name] = run_task_evaluation(
            task_name=task_name,
            max_steps=max_steps,
            verbose=verbose
        )
    
    # Compute summary with success metrics
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_steps": sum(r["steps_taken"] for r in results.values()),
        "total_reward": sum(r["total_reward"] for r in results.values()),
        "avg_score": sum(r["task_score"] for r in results.values()) / 3,
        "avg_iou": sum(r["final_iou"] for r in results.values()) / 3,
        "success_rate": sum(1 for task, r in results.items() if r["final_iou"] >= task_targets[task]) / 3,
        "task_results": {
            task: {
                "final_iou": results[task]["final_iou"],
                "task_score": results[task]["task_score"],
                "steps": results[task]["steps_taken"],
                "reward": results[task]["total_reward"]
            }
            for task in results
        }
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("[SUMMARY] Evaluation Results")
        print(f"{'='*70}")
        print(f"Average Task Score:  {summary['avg_score']:.3f}")
        print(f"Average IoU:         {summary['avg_iou']:.4f}")
        print(f"Success Rate:        {summary['success_rate']:.1%}")
        print(f"Total Steps:         {summary['total_steps']}")
        print(f"Total Reward:        {summary['total_reward']:.2f}")
        print()
        for task, stats in summary["task_results"].items():
            print(f"{task.upper():10} | Score: {stats['task_score']:.3f} | "
                  f"IoU: {stats['final_iou']:.4f} | Steps: {stats['steps']:2d} | "
                  f"Reward: {stats['reward']:+.2f}")
        print(f"{'='*70}\n")
    
    return {"summary": summary, "detailed_results": results}


if __name__ == "__main__":
    # Run with heuristic agent if no OpenAI API
    if not os.environ.get("OPENAI_API_KEY"):
        print("[INFO] No OPENAI_API_KEY found, using heuristic agent")
    
    # Run evaluation
    results = evaluate_all_tasks(max_steps=20, verbose=True)
    
    # Save results to JSON
    output_file = "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results saved to {output_file}")
