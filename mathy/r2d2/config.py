from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class MathyArgs(BaseModel):
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = None
    model_dir: str = "/tmp/r2d2-training/"
    model_name: str = "model.h5"
    units: int = 64
    # Units for math embeddings
    embedding_units: int = 128
    init_model_from: Optional[str] = None
    train: bool = False
    verbose: bool = False
    lr: float = 3e-4

    actor_update_from_learner_every_n: int = 50

    replay_size: int = 8192
    replay_ready: int = 4096
    max_eps: int = 25000
    gamma: float = 0.99
    exploration_greedy_epsilon: float = 0.01
    # Worker's sleep this long between steps to allow
    # other workers time to process. This is useful for
    # running more workers than you have processors to
    # get a better diversity of experience.
    actor_timestep_wait: float = 0.5
    # The number of worker agents to create.
    num_actors: int = 24
    num_learners: int = 1

    # When profile is true, each A3C worker thread will output a .profile
    # file in the model save path when it exits.
    profile: bool = False

    # Whether to use the reward prediction aux task
    use_reward_prediction = True

    # Whether to use the value replay aux task
    use_value_replay = True

    # Whether to use the grouping change aux task
    use_grouping_control = True
