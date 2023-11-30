from unittest.mock import patch

import pytest
from click.testing import CliRunner
from mathy.cli import cli
from mathy_envs.env import MathyEnv
from mathy_envs.gym import MathyGymEnv
from mathy_envs.gym.mathy_gym_env import safe_register
from mathy_envs.types import MathyEnvProblem, MathyEnvProblemArgs


class InvalidProblemEnv(MathyEnv):
    def get_env_namespace(self) -> str:
        return "mathy.binomials.mulptiply"

    def problem_fn(self, params: MathyEnvProblemArgs) -> MathyEnvProblem:
        return MathyEnvProblem("4++++++7", 1, self.get_env_namespace())


class InvalidProblemGymEnv(MathyGymEnv):
    def __init__(self, **kwargs):
        super(InvalidProblemGymEnv, self).__init__(
            env_class=InvalidProblemEnv, **kwargs
        )


def test_cli_contribute():
    runner = CliRunner()

    with patch("webbrowser.open") as mock:
        result = runner.invoke(cli, ["contribute"])
        mock.assert_called_with("https://mathy.ai/contributing", new=2)
        assert result.exit_code == 0


def test_cli_problems():
    runner = CliRunner()
    for problem_type in ["poly", "poly-combine", "complex", "binomial"]:
        result = runner.invoke(cli, ["problems", problem_type, "--number=100"])
        assert result.exit_code == 0


def test_cli_problems_parse_error():
    safe_register(
        id="mathy-invalid-easy-v0", entry_point="tests.test_cli:InvalidProblemGymEnv",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["problems", "invalid", "--number=100"])
    print(result.stdout)
    assert result.exit_code == 0


def test_cli_simplify():
    runner = CliRunner()
    for problem in ["4x + 2x"]:
        result = runner.invoke(cli, ["simplify", problem, "--max-steps=10"])
        assert result.exit_code == 0


@pytest.mark.parametrize("use_mp", [True, False])
def test_cli_simplify_swarm(use_mp: bool):
    runner = CliRunner()
    args = ["simplify", "4x + 2x"]
    if not use_mp:
        args.append("--single-process")
    result = runner.invoke(cli, args)
    assert result.exit_code == 0
