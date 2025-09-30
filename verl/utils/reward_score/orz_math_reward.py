# A thin wrapper that lets us reuse VERL's MATH reward on custom data_source names.
from verl.utils.reward_score.math_verify import compute_score as compute_score_math_verify
import re
    
# def format_reward(completions, **kwargs):
#     """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
#     pattern = r"^<think>\n.*?\n</think>\n.*?\n<answer>\n.*?\n</answer>$"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]


def compute_score(*, data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Ignore data_source and delegate to the built-in MATH scorer.
    # The built-in returns either a float or {"score": float, ...}; both are accepted.
    return compute_score_math_verify(model_output=solution_str,
                                    ground_truth=ground_truth)
