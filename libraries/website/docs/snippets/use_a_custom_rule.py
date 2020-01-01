#!pip install gym
import os
import shutil
import tempfile

import gym

from mathy import (
    AddExpression,
    BaseRule,
    MathExpression,
    ExpressionParser,
    MathyEnv,
    NegateExpression,
    SubtractExpression,
)
from mathy.envs import PolySimplify
from mathy.agents.a3c import A3CAgent, A3CConfig
from mathy.cli import setup_tf_env


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
all_rules = MathyEnv.core_rules() + [PlusNegationRule()]
# Specify a set of operators to choose from when generating poly simplify problems
env_args = {"ops": ["+", "-"], "rules": all_rules}
# Configure and launch the A3C agent training
args = A3CConfig(
    max_eps=1,
    num_workers=1,
    print_training=True,
    topics=["poly"],
    model_dir=model_folder,
)
A3CAgent(args, env_extra=env_args).train()
# Comment this out to keep your model
shutil.rmtree(model_folder)
