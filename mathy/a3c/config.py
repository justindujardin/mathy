from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MathyGymEnvTypes(str, Enum):
    poly_easy = "mathy-poly-easy-v0"
    poly_normal = "mathy-poly-normal-v0"
    poly_hard = "mathy-poly-hard-v0"

    poly_blockers_easy = "mathy-poly-blockers-easy-v0"
    poly_blockers_normal = "mathy-poly-blockers-normal-v0"
    poly_blockers_hard = "mathy-poly-blockers-hard-v0"

    binomial_easy = "mathy-binomial-easy-v0"
    binomial_normal = "mathy-binomial-normal-v0"
    binomial_hard = "mathy-binomial-hard-v0"


class A3CAgentTypes(str, Enum):
    a3c = "a3c"
    random = "random"


class A3CArgs(BaseModel):
    env_name: MathyGymEnvTypes = MathyGymEnvTypes.poly_blockers_easy
    algorithm: A3CAgentTypes = A3CAgentTypes.a3c
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model.h5"
    units: int = 128
    init_model_from: Optional[str] = None
    train: bool = False
    lr: float = 3e-4
    update_freq: int = 50
    max_eps: int = 10000
    gamma: float = 0.99
    # Worker's sleep this long between steps to allow
    # other threads time to process. This is useful for
    # running more threads than you have processors to
    # get a better diversity of experience.
    worker_wait: float = 0.01
    # The number of worker agents to create.
    num_workers: int = 3

    # 0-1 controlling how often to choose a random action rather than
    # use the predicted policy. e.g. 0.1 would choose random actions
    # 10% of the time.
    exploration_greedy_epsilon: float = 0.05
    # `H` term from the A3C paper, controls the scaling of the entroy
    # of the policy before adding to the total loss. This encourages
    # exploration.
    entropy_beta: float = 0.01
