from dataclasses import dataclass

from fragile.core.swarm import Swarm

from .solver import SwarmConfig, swarm_solve


@dataclass
class MathyAPISwarmState:
    config: SwarmConfig


class Mathy:
    """The standard interface for working with Mathy models and agents."""

    state: MathyAPISwarmState

    def __init__(
        self, *, config: SwarmConfig = None, silent: bool = False,
    ):
        if config is None:
            config = SwarmConfig()
        if not isinstance(config, SwarmConfig):
            raise ValueError("config must be a SwarmConfig instance")
        self.state = MathyAPISwarmState(config=config)

    def simplify(self, *, problem: str, max_steps: int = None) -> Swarm:
        if max_steps is not None:
            return swarm_solve(problem, self.state.config, max_steps=max_steps)
        return swarm_solve(problem, self.state.config)
