from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class A3CArgs(BaseModel):
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = None
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model.h5"
    units: int = 128
    embedding_units: int = 128
    lstm_units: int = 128
    init_model_from: Optional[str] = None
    train: bool = False
    verbose: bool = False
    # The history size for the greedy worker
    greedy_history_size: int = 4096
    # History size for exploratory workers
    history_size: int = 2048
    # Size at which it's okay to start sampling from the memory
    ready_at: int = 1024
    lr: float = 3e-4
    update_freq: int = 25
    max_eps: int = 100000
    gamma: float = 0.99

    entropy_loss_scaling = 0.25

    e_greedy_min = 0.001
    e_greedy_max = 0.4
    # Worker's sleep this long between steps to allow
    # other threads time to process. This is useful for
    # running more threads than you have processors to
    # get a better diversity of experience.
    worker_wait: float = 0.5
    # The number of worker agents to create.
    num_workers: int = 3

    # When profile is true, each A3C worker thread will output a .profile
    # file in the model save path when it exits.
    profile: bool = False

    # Whether to use the reward prediction aux task
    use_reward_prediction = False

    # Whether to use the value replay aux task
    use_value_replay = False

    # Whether to use the grouping change aux task
    use_grouping_control = True
