from mathy_envs import MathyEnv, MathyEnvProblem, MathyEnvProblemArgs


class CustomSimplifyEnv(MathyEnv):
    def get_env_namespace(self) -> str:
        return "custom.polynomial.simplify"

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        return MathyEnvProblem("4x + y + 13x", 3, self.get_env_namespace())


env: MathyEnv = CustomSimplifyEnv()
state, problem = env.get_initial_state()
assert problem.text == "4x + y + 13x"
assert problem.complexity == 3
