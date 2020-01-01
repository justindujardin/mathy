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
    mathy_version: str = about.__version__

    units: int = 64
    embedding_units: int = 128
    lstm_units: int = 128
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = None
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model"
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

    # How naturally ordered are the terms in the expression?
    use_term_order = False
    # How much internal branching complexity is in the tree?
    use_tree_complexity = False
    # Whether to use the grouping change aux task
    use_grouping_control = False
    # Clip signal at 0.0 so it does not optimize into the negatives
    clip_grouping_control = False

    # Include the time/type environment features in the embeddings
    use_env_features = False

    # Include the node values floating point features in the embeddings
    use_node_values = True
