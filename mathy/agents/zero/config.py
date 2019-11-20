from ...agents.base_config import BaseConfig


class SelfPlayConfig(BaseConfig):
    batch_size = 32
    epochs = 3
    mcts_sims: int = 50
    temperature_threshold: float = 0.5
    self_play_problems: int = 12
    cpuct: float = 1.0
