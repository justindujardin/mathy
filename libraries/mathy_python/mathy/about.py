__title__ = "mathy"
__version__ = "0.7.14"
__summary__ = "Mathy - RL environments for solving math problems step-by-step"
__uri__ = "https://mathy.ai"
__author__ = "Justin DuJardin"
__email__ = "justin@dujardinconsulting.com"
__license__ = "All rights reserved"


class PackageExtras:
    AGENTS: str = "agents"
    SOLVER: str = "solver"
    REFORMER: str = "reformer"

    @staticmethod
    def requires(extras_name: str):
        try:
            if extras_name == PackageExtras.SOLVER:
                import gym
                import fragile
            elif extras_name == PackageExtras.AGENTS:
                import gym
                import tensorflow
            elif extras_name == PackageExtras.REFORMER:
                import torch
                import reformer_pytorch
            else:
                raise ValueError(
                    f"The provided extras requirement is not known: {extras_name}"
                )
        except ImportError:
            alert = (
                "\n\nThe functionality you are trying to use requires optional "
                "packages that you don't have installed. Try running:\n\n"
                f"\tpip install mathy[{extras_name}]\n\n"
            )
            raise ValueError(alert)
