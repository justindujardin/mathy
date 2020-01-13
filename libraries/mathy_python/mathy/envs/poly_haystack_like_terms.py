from typing import Any, Dict, List, Optional, Type, Union

from numpy import random
from numpy.random import randint

from .. import time_step
from ..core.expressions import MathExpression
from ..util import TermEx, get_term_ex, get_terms
from ..env import MathyEnvProblem
from ..problems import (
    gen_combine_terms_in_place,
    get_rand_term_templates,
    get_rand_vars,
    mathy_term_string,
    maybe_power,
    rand_bool,
    rand_op,
    rand_var,
    split_in_two_random,
)
from ..core.rule import BaseRule
from ..rules import (
    AssociativeSwapRule,
    CommutativeSwapRule,
    ConstantsSimplifyRule,
    DistributiveFactorOutRule,
    DistributiveMultiplyRule,
    VariableMultiplyRule,
)
from ..state import MathyEnvState, MathyEnvStateStep, MathyObservation
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs, EnvRewards
from .poly_simplify import PolySimplify


class PolyHaystackLikeTerms(PolySimplify):
    """Act on any node in the expression that has another term like it
    somewhere else. For example in the problem:

    2x + 8 + 13.2y + z^2 + 5x
    ^^---------------------^^

    Applying any rule to one of those nodes is a win. The idea here is that
    in order to succeed at this task, the model must build a representation
    that can identify like terms in a large expression tree.
    """

    def __init__(self, **kwargs):
        super(PolyHaystackLikeTerms, self).__init__(**kwargs)

    def get_env_namespace(self) -> str:
        return "mathy.polynomials.haystack.like.terms"

    def get_penalizing_actions(self, state: MathyEnvState) -> List[Type[BaseRule]]:
        return [
            CommutativeSwapRule,
            AssociativeSwapRule,
            DistributiveFactorOutRule,
            DistributiveMultiplyRule,
            ConstantsSimplifyRule,
            VariableMultiplyRule,
        ]

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        return problem.complexity

    def transition_fn(
        self,
        env_state: MathyEnvState,
        expression: MathExpression,
        features: MathyObservation,
    ) -> Optional[time_step.TimeStep]:
        """If all like terms are siblings."""
        agent = env_state.agent
        if len(agent.history) == 0:
            return None
        # History gets pushed before this fn, so history[-1] is the current state,
        # and history[-2] is the previous state. Find the previous state node we
        # acted on, and compare to that.
        curr_timestep: MathyEnvStateStep = agent.history[-1]
        last_timestep: MathyEnvStateStep = agent.history[-2]
        expression = self.parser.parse(last_timestep.raw)
        action_node = self.get_token_at_index(expression, curr_timestep.focus)
        touched_term = get_term_ex(action_node)

        term_nodes = get_terms(expression)
        # We have the token_index of the term that was acted on, now we have to see
        # if that term has any like siblings (not itself). We do this by ignoring the
        # term with a matching r_index to the node the agent acted on.
        #
        # find_nodes updates the `r_index` value on each node which is the token index
        BaseRule().find_nodes(expression)

        like_counts: Dict[str, int] = {}
        all_indices: Dict[str, List[int]] = {}
        max_index = 0
        for term_node in term_nodes:
            max_index = max(max_index, term_node.r_index)
            ex: Optional[TermEx] = get_term_ex(term_node)
            if ex is None:
                continue

            key = mathy_term_string(variable=ex.variable, exponent=ex.exponent)
            if key == "":
                key = "const"
            if key not in like_counts:
                like_counts[key] = 1
            else:
                like_counts[key] += 1
            if key not in all_indices:
                all_indices[key] = [term_node.r_index]
            else:
                all_indices[key].append(term_node.r_index)

        like_indices: Optional[List[int]] = None
        for key in all_indices.keys():
            if len(all_indices[key]) > 1:
                like_indices = all_indices[key]
        if action_node is not None and touched_term is not None:
            touched_key = mathy_term_string(
                variable=touched_term.variable, exponent=touched_term.exponent
            )
            if touched_key in like_counts and like_counts[touched_key] > 1:
                action_node.all_changed()
                return time_step.termination(features, self.get_win_signal(env_state))

        if env_state.agent.moves_remaining <= 0:
            distances = []
            if like_indices is not None:
                for index in like_indices:
                    distances.append(abs(index - action_node.r_index))
                loss_magnitude = min(distances) / max_index
            else:
                loss_magnitude = 1.0
            lose_signal = EnvRewards.LOSE - loss_magnitude
            return time_step.termination(features, lose_signal)
        return None

    def make_problem(
        self,
        min_terms: int,
        max_terms: int,
        like_terms: int,
        exponent_probability: float,
    ) -> str:
        assert min_terms <= max_terms, "min cannot be greater than max"
        assert like_terms < min_terms, "must have atleast one term that is not like"
        out_terms = []
        total_terms = random.randint(min_terms, max_terms)
        num_diff_terms = total_terms - like_terms
        diff_term_tpls = get_rand_term_templates(
            num_diff_terms + 1, exponent_probability=exponent_probability
        )
        like_term_tpl = diff_term_tpls[-1]
        diff_term_tpls = diff_term_tpls[:-1]

        for i in range(like_terms):
            out_terms.append(like_term_tpl.make())

        for tpl in diff_term_tpls:
            out_terms.append(tpl.make())
        random.shuffle(out_terms)
        problem = f" + ".join(out_terms)
        return problem
        # return "5i + i + 9i^2"

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        if params.difficulty == MathyEnvDifficulty.easy:
            text = self.make_problem(
                min_terms=3, max_terms=8, like_terms=2, exponent_probability=0.3
            )
        elif params.difficulty == MathyEnvDifficulty.normal:
            text = self.make_problem(
                min_terms=4, max_terms=7, like_terms=2, exponent_probability=0.5
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            text = self.make_problem(
                min_terms=5, max_terms=12, like_terms=2, exponent_probability=0.4
            )
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, 2, self.get_env_namespace())
