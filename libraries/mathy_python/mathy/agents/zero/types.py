from pydantic import BaseModel
from typing import NamedTuple, Any
from ...state import MathyObservation


class EpisodeSummary(BaseModel):
    text: str
    complexity: int
    duration: float
    reward: float
    solved: bool


class EpisodeHistory(NamedTuple):
    """A tuple of items related to an episode timestep.
    - "text" The source text at the state
    - "action" The action taken
    - "reward" The undiscounted reward observed
    - "discounted" The final discounted reward observed
    - "terminal" 0/1 whether this state was terminal
    - "observation" MathyObservation for training
    - "pi" weighted distribution over the actions
    - "value" predicted value for the current state
    """

    text: str
    action: int
    reward: float
    discounted: float
    terminal: bool
    observation: MathyObservation
    pi: Any
    value: float
