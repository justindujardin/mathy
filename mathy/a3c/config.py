from typing import Optional
from pydantic import BaseModel, Schema
from enum import Enum


class MathyGymEnvTypes(str, Enum):
    poly03 = "mathy-poly-03-v0"
    poly04 = "mathy-poly-04-v0"
    poly05 = "mathy-poly-05-v0"
    poly06 = "mathy-poly-06-v0"
    poly07 = "mathy-poly-07-v0"
    poly8 = "mathy-poly-08-v0"
    poly9 = "mathy-poly-09-v0"
    poly10 = "mathy-poly-10-v0"

    binomial_easy = "mathy-binomial-easy-v0"
    binomial_normal = "mathy-binomial-normal-v0"
    binomial_hard = "mathy-binomial-hard-v0"


class A3CAgentTypes(str, Enum):
    a3c = "a3c"
    random = "random"


class A3CArgs(BaseModel):
    env_name: MathyGymEnvTypes = MathyGymEnvTypes.poly03
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
