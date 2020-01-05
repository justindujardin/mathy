from ...agents.base_config import BaseConfig


class SelfPlayConfig(BaseConfig):
    batch_size = 128
    epochs = 10
    mcts_sims: int = 50
    temperature_threshold: float = 0.5
    self_play_problems: int = 64
    training_iterations: int = 100
    cpuct: float = 1.0


    # When profile is true and workers == 1 the main thread will output worker_0.profile on exit
    profile: bool = False
