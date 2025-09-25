# A thin wrapper that lets us reuse VERL's MATH reward on custom data_source names.
from verl.utils.reward_score.math_verify import compute_score as compute_score_math_verify

def compute_score(*, data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Ignore data_source and delegate to the built-in MATH scorer.
    # The built-in returns either a float or {"score": float, ...}; both are accepted.
    return compute_score_math_verify(model_output=solution_str,
                                    ground_truth=ground_truth)
