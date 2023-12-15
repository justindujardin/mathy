from dataclasses import dataclass
from typing import Optional

from .fragile.core.swarm import Swarm

from .solver import SwarmConfig, swarm_solve


@dataclass
class MathyAPISwarmState:
    config: SwarmConfig


class Mathy:
    """The standard interface for working with Mathy models and agents."""

    state: MathyAPISwarmState

    def __init__(
        self,
        *,
        config: Optional[SwarmConfig] = None,
        silent: bool = False,
    ):
        self.silent = silent
        if config is None:
            config = SwarmConfig()
        if not isinstance(config, SwarmConfig):
            raise ValueError("config must be a SwarmConfig instance")
        self.state = MathyAPISwarmState(config=config)

    def simplify(self, *, problem: str, max_steps: Optional[int] = None) -> Swarm:
        if max_steps is not None:
            return swarm_solve(
                problem, self.state.config, max_steps=max_steps, silent=self.silent
            )
        return swarm_solve(problem, self.state.config, silent=self.silent)
