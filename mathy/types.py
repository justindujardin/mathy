from typing import List, Any, NamedTuple


class MathyEnvProblem(NamedTuple):
    """Summarize an environment-specific problem that was generated with
    a tuple of (text, complexity, type) where:
     - "text" is the text content of the generated problem
     - "complexity" is an integer value that represents the number of
       terms in the problem text.
     - "type" is an integer value representing the problem type that
       the environment generates.
    """

    text: str
    complexity: int
    type: int


class MathyEnvObservation(NamedTuple):
    """Summarize an environment observation in a named tuple."""

    input: str
    output: str
    action: int
    token: int
    reward: float
    discounted: float
    policy: List[float]
    features: Any
    problem: str


class MathyEnvEpisodeResult(NamedTuple):
    """Summarize episode history and final result."""

    history: List[MathyEnvObservation]
    episode_reward: float
    is_win: bool
