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
    """A tuple of items related to an episode timestep."""

    text: str
    action: int
    reward: float
    discounted: float
    terminal: bool
    observation: MathyObservation
    pi: Any
    value: float


# fmt: off
EpisodeHistory.text.__doc__ = "The problem text at this state" # noqa
EpisodeHistory.action.__doc__ = "The action taken" # noqa
EpisodeHistory.reward.__doc__ = "The undiscounted reward observed" # noqa
EpisodeHistory.discounted.__doc__ = "The final discounted reward observed" # noqa
EpisodeHistory.terminal.__doc__ = "0/1 whether this state was terminal" # noqa
EpisodeHistory.observation.__doc__ = "MathyObservation for training" # noqa
EpisodeHistory.pi.__doc__ = "weighted distribution over the actions" # noqa
EpisodeHistory.value.__doc__ = "predicted value for the current state" # noqa
# fmt: on
