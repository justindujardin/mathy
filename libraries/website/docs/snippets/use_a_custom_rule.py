#!pip install gym
import shutil
import tempfile
from typing import List

from mathy.agent import A3CAgent, AgentConfig
from mathy.cli import setup_tf_env
from mathy_core import (
    AddExpression,
    BaseRule,
    MathExpression,
    NegateExpression,
    SubtractExpression,
)
from mathy_envs import MathyEnv


class PlusNegationRule(BaseRule):
    """Convert subtract operators to plus negative to allow commuting"""

    @property
    def name(self) -> str:
        return "Plus Negation"

    @property
    def code(self) -> str:
        return "PN"

    def can_apply_to(self, node: MathExpression) -> bool:
        is_sub = isinstance(node, SubtractExpression)
        is_parent_add = isinstance(node.parent, AddExpression)
        return is_sub and (node.parent is None or is_parent_add)

    def apply_to(self, node: MathExpression):
        change = super().apply_to(node)
        change.save_parent()  # connect result to node.parent
        result = AddExpression(node.left, NegateExpression(node.right))
        result.set_changed()  # mark this node as changed for visualization
        return change.done(result)


# Quiet tensorflow debug outputs
setup_tf_env()
# Train in a temporary folder
model_folder = tempfile.mkdtemp()
# Add an instance of our new rule to the built-int environment rules
all_rules: List[BaseRule] = MathyEnv.core_rules() + [PlusNegationRule()]
# Specify a set of operators to choose from when generating poly simplify problems
env_args = {"ops": ["+", "-"], "rules": all_rules}
# Configure and launch the A3C agent training
args = AgentConfig(
    max_eps=1,
    num_workers=1,
    print_training=True,
    topics=["poly"],
    model_dir=model_folder,
)
A3CAgent(args, env_extra=env_args).train()
# Comment this out to keep your model
shutil.rmtree(model_folder)
