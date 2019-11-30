"""Environment with user-defined actions"""

from mathy import (
    AddExpression,
    BaseRule,
    MathyEnvState,
    NegateExpression,
    SubtractExpression,
    envs,
    mathy_core_rules,
)


class PlusNegationRule(BaseRule):
    """Convert subtract operators to plus negative to allow commuting"""

    @property
    def name(self) -> str:
        return "Plus Negation"

    @property
    def code(self) -> str:
        return "PN"

    def can_apply_to(self, node) -> bool:
        is_sub = isinstance(node, SubtractExpression)
        is_parent_add = isinstance(node.parent, AddExpression)
        return is_sub and (node.parent is None or is_parent_add)

    def apply_to(self, node):
        change = super().apply_to(node)
        change.save_parent()  # connect result to node.parent
        result = AddExpression(node.left, NegateExpression(node.right))
        result.set_changed()  # mark this node as changed for visualization
        return change.done(result)


class CustomActionEnv(envs.PolySimplify):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rules = mathy_core_rules() + [PlusNegationRule()]


env = CustomActionEnv()

state = MathyEnvState(problem="4x - 2x")
expression = env.parser.parse(state.agent.problem)
action = env.random_action(expression, PlusNegationRule)
out_state, transition, _ = env.get_next_state(state, action)
assert out_state.agent.problem == "4x + -2x"
