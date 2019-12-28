from ..env import MathyEnvProblem
from ..problems import gen_combine_terms_in_place, rand_bool
from ..types import MathyEnvDifficulty, MathyEnvProblemArgs
from .poly_simplify import PolySimplify


class PolyCombineInPlace(PolySimplify):
    """A Mathy environment for combining like terms in-place without
    any commuting. This task is intended to test the model's ability
    to identify like-terms among a bunch of unlike terms and combine
    them with a sequence of two moves.
    """

    def get_env_namespace(self) -> str:
        return "mathy.polynomials.combine.in.place"

    def max_moves_fn(
        self, problem: MathyEnvProblem, config: MathyEnvProblemArgs
    ) -> int:
        """When combining terms that are already siblings, we only need
        to take two actions:

            1. distributive factor out the common element
            2. simplify the remaining constants

         """

        return 2

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        if params.difficulty == MathyEnvDifficulty.easy:
            powers = rand_bool(20)
            easy = rand_bool(65)
            text, _ = gen_combine_terms_in_place(
                min_terms=6, max_terms=12, easy=easy, powers=powers
            )
        elif params.difficulty == MathyEnvDifficulty.normal:
            powers = rand_bool(40)
            easy = rand_bool(15)
            text, _ = gen_combine_terms_in_place(
                min_terms=9, max_terms=16, easy=easy, powers=powers
            )
        elif params.difficulty == MathyEnvDifficulty.hard:
            powers = rand_bool(60)
            easy = rand_bool(5)
            text, _ = gen_combine_terms_in_place(
                min_terms=16, max_terms=22, easy=easy, powers=powers
            )
        else:
            raise ValueError(f"Unknown difficulty: {params.difficulty}")
        return MathyEnvProblem(text, 2, self.get_env_namespace())
