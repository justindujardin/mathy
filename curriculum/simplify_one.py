from ..core.problems import MODE_SIMPLIFY_POLYNOMIAL, ProblemGenerator


config = {
    "problem_types": [MODE_SIMPLIFY_POLYNOMIAL],
    "mcts_sims": 500,
    "max_turns": 15,
    "problem_fn": lambda: ProblemGenerator().
}
