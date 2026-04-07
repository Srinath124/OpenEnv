"""
ML inference engine using OpenAI API for agent decision-making.
Evaluates agent performance on OffroadSegNet ML environment across all task difficulties.
"""

import os
import json
from typing import Optional
from openai import OpenAI
from environment import Environment
from models import Action
from graders import EasyTaskGrader, MediumTaskGrader, HardTaskGrader


class MLAgentInference:
    """Uses OpenAI API to drive ML engineering decisions."""
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 100):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.model = model
        self.max_tokens = max_tokens
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None
    
    def get_action(self, observation_report: str) -> Optional[str]:
        """
        Query OpenAI API for next action given experiment report.
        Returns action name or None if API unavailable.
        """
        if not self.client:
            return None
        
        try:
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
                temperature=0.7,
            )
            
            content = response.choices[0].message.content
            action_dict = json.loads(content)
            return action_dict.get("action")
        
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"API Error: {e}")
            return None
    
    def heuristic_action(self, observation_report: str) -> str:
        """
        Fallback heuristic when API is unavailable.
        Simple rule-based strategy for debugging.
        """
        if "instability" in observation_report.lower():
            return "adjust_learning_rate"
        elif "overfitting" in observation_report.lower():
            return "increase_regularization"
        elif "Sky" in observation_report and "0.00" in observation_report:
            return "enable_class_balancing"
        elif "Generalization" in observation_report.lower():
            return "enable_augmentation"
        else:
            return "run_training_epoch"


def run_task_evaluation(task_name: str = "easy", max_steps: int = 15) -> dict:
    """
    Run complete task evaluation with OpenAI agent or heuristic fallback.
    Returns: final state, task score, trajectory statistics
    """
    print(f"\n{'='*70}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'='*70}")
    
    # Task-specific targets
    targets = {
        "easy": {"target_iou": 0.50, "description": "Improve IoU to acceptable level"},
        "medium": {"target_iou": 0.55, "description": "Improve IoU while preventing overfitting"},
        "hard": {"target_iou": 0.65, "description": "Multi-objective: high IoU + sky coverage + efficiency"}
    }
    
    target_info = targets.get(task_name, targets["easy"])
    
    env = Environment(target_iou=target_info["target_iou"], seed=42)
    agent = MLAgentInference()
    
    obs = env.reset()
    print(f"\nTarget: {target_info['description']}")
    print(f"Target IoU: {target_info['target_iou']}")
    print(f"\nInitial state: Mean IoU = {obs.mean_iou:.4f}")
    print(f"Compute budget: {obs.compute_remaining_minutes} minutes")
    
    trajectory = []
    total_reward = 0.0
    step_count = 0
    
    for step in range(max_steps):
        # Get action from API or heuristic
        action_name = agent.get_action(obs.report)
        
        if not action_name:
            # Fallback to heuristic
            action_name = agent.heuristic_action(obs.report)
        
        print(f"\n[Step {step+1}] Action: {action_name}")
        
        # Execute action
        try:
            action = Action(action=action_name)
            obs, reward, done, info = env.step(action)
            
            step_count += 1
            total_reward += reward.value
            
            trajectory.append({
                "step": step + 1,
                "action": action_name,
                "mean_iou": obs.mean_iou,
                "reward": reward.value,
                "message": info.get("message", ""),
            })
            
            # Print progress
            print(f"  IoU: {obs.mean_iou:.4f} | Reward: {reward.value:+.2f} | Budget: {obs.compute_remaining_minutes}m")
            
            if done:
                print(f"\nEpisode complete after {step_count} steps")
                break
        
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    # Compute final score
    state = env.state()
    if task_name == "easy":
        score = EasyTaskGrader.score(state)
    elif task_name == "medium":
        score = MediumTaskGrader.score(state)
    else:
        score = HardTaskGrader.score(state)
    
    result = {
        "task": task_name,
        "final_mean_iou": state["mean_iou"],
        "final_sky_iou": state["sky_iou"],
        "task_score": score,
        "total_reward": total_reward,
        "steps": step_count,
        "trajectory": trajectory,
    }
    
    return result


def print_evaluation_summary(results: list):
    """Print comprehensive evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for result in results:
        print(f"\n📊 Task: {result['task'].upper()}")
        print(f"  Task Score:       {result['task_score']:.4f}/1.0")
        print(f"  Final IoU:        {result['final_mean_iou']:.4f}")
        print(f"  Sky IoU:          {result['final_sky_iou']:.4f}")
        print(f"  Total Reward:     {result['total_reward']:+.2f}")
        print(f"  Steps Taken:      {result['steps']}")
    
    # Calculate averages
    avg_score = sum(r["task_score"] for r in results) / len(results)
    avg_iou = sum(r["final_mean_iou"] for r in results) / len(results)
    
    print("\n" + "-"*70)
    print(f"Average Task Score: {avg_score:.4f}/1.0")
    print(f"Average IoU:        {avg_iou:.4f}")
    print("="*70)


def main():
    """Main inference entry point."""
    print("\n🚀 OffroadSegNet ML Engineering Environment - Agent Evaluation")
    print("Using OpenAI API (or heuristic fallback)\n")
    
    # Run evaluation on all tasks
    results = []
    for task in ["easy", "medium", "hard"]:
        result = run_task_evaluation(task, max_steps=15)
        results.append(result)
    
    # Print summary
    print_evaluation_summary(results)
    
    # Per-task interpretation
    print("\n📝 RESULTS INTERPRETATION:")
    print("-"*70)
    print("""
Easy Task (Score 0.00 - 1.00):
  - 0.00-0.30: Poor debugging (wrong configuration choices)
  - 0.30-0.70: Partial improvement (missing key actions)
  - 0.70-1.00: Good improvement (reached target > 0.50 IoU)

Medium Task (Score 0.00 - 1.00):
  - Requires avoiding overfitting while improving IoU
  - Optimal: Enable regularization and augmentation
  - 0.70+: Excellent balance of performance and generalization

Hard Task (Score 0.00 - 1.00):
  - Multi-objective: IoU + Sky coverage + Efficiency + Stability
  - 0.40-0.70: Good partial progress on objectives
  - 0.80+: Expert-level multi-objective optimization
    """)


if __name__ == "__main__":
    main()

