from typing import List, Optional

from pydantic import BaseModel
from .. import about


class BaseConfig(BaseModel):
    class Config:
        extra = "allow"

    name: str = "unnamed_model"
    # The version of the model. You probably don't want to change this.
    version: str = "0.0.1"
    description: str = "Mathy.ai trained model"
    license: str = "CC BY-SA 3.0"
    author: str = about.__author__
    email: str = about.__email__
    url: str = about.__uri__
    mathy_version: str = f">={about.__version__},<1.0.0"
    # One of "batch" or "layer"
    normalization_style = "layer"
    # The number of timesteps use when making predictions. This includes the current timestep and
    # (n - 1) previous timesteps
    prediction_window_size: int = 16
    # Dropout to apply to LSTMs
    dropout: float = 0.2
    units: int = 64
    embedding_units: int = 128
    lstm_units: int = 128
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = None
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model"
    init_model_from: Optional[str] = None
    verbose: bool = False
    # Initial learning rate that decays over time.
    lr_initial: float = 0.01
    lr_decay_steps: int = 100
    lr_decay_rate: float = 0.96
    lr_decay_staircase: bool = True
    max_eps: int = 15000
    # How often to write histograms to tensorboard (in training steps)
    summary_interval: int = 100
    gamma: float = 0.99
    # The number of worker agents to create.
    num_workers: int = 3
    # The lambda value for generalized lambda returns to calculate value loss
    # 0.0 = bootstrap values, 1.0 = discounted
    td_lambda: float = 0.2
    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False
    # This is very verbose and prints every policy_value_model.call time
    print_model_call_times: bool = False
    # Print mode for output. "terminal" is the default, also supports "attention"
    # NOTE: attention is gone (like... the layer)
    print_mode: str = "terminal"
