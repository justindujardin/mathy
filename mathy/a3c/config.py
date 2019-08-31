from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MathyGymEnvTypes(str, Enum):
    poly_easy = "mathy-poly-easy-v0"
    poly_normal = "mathy-poly-normal-v0"
    poly_hard = "mathy-poly-hard-v0"

    complex_easy = "mathy-complex-easy-v0"
    complex_normal = "mathy-complex-normal-v0"
    complex_hard = "mathy-complex-hard-v0"

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
    update_freq: int = 25
    max_eps: int = 10000
    gamma: float = 0.99
    # Worker's sleep this long between steps to allow
    # other threads time to process. This is useful for
    # running more threads than you have processors to
    # get a better diversity of experience.
    worker_wait: float = 0.01
    # The number of worker agents to create.
    num_workers: int = 3

    # When profile is true, each A3C worker thread will output a .profile
    # file in the model save path when it exits.
    profile: bool = False
