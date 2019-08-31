from typing import List, Any, NamedTuple
from enum import Enum
from pydantic import BaseModel, Schema


class MathyEnvDifficulty(str, Enum):
    easy: str = Schema(  # type:ignore
        default="easy",
        title="Easy Problems",
        description="The simplest form of problems that demonstrate a task",
    )
    normal: str = Schema(  # type:ignore
        default="normal",
        title="Normal Problems",
        description="Challenging problems that involve more terms",
    )
    hard: str = Schema(  # type:ignore
        default="hard",
        title="Hard Problems",
        description="Difficult problems that have intentionally large expression trees.",
    )


class MathyEnvProblemArgs(BaseModel):
    difficulty: MathyEnvDifficulty = MathyEnvDifficulty.easy
    turns_per_complexity: int = 4


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
