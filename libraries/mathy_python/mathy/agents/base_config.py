from typing import List, Optional

from pydantic import BaseModel


class BaseConfig(BaseModel):
    units: int = 64
    embedding_units: int = 512
    lstm_units: int = 256
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = None
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model"
    model_format: str = "keras"
    init_model_from: Optional[str] = None
    train: bool = False
    verbose: bool = False
    lr: float = 3e-4
    max_eps: int = 15000
    # How often to write histograms to tensorboard (in training steps)
    summary_interval: int = 100
    gamma: float = 0.99
    # The number of worker agents to create.
    num_workers: int = 3
    # The lambda value for generalized lambda returns to calculate value loss
    # 0.0 = bootstrap values, 1.0 = discounted
    td_lambda: float = 0.3
    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False
    # Print mode for output. "terminal" is the default, also supports "attention"
    # NOTE: attention is gone (like... the layer)
    print_mode: str = "terminal"

