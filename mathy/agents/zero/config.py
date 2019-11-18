from ...agents.base_config import BaseConfig


class SelfPlayConfig(BaseConfig):
    batch_size = 64
    epochs = 10
    mcts_sims: int = 150
    num_mcts_sims: int = 15
    temperature_threshold: float = 0.5
    self_play_problems: int = 100
    cpuct: float = 1.0
