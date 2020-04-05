import shutil
import tempfile
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from mathy.cli import cli
from mathy.envs.gym import MathyGymEnv  # noqa


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


def test_cli_simplify():
    runner = CliRunner()
    for problem in ["4x + 2x", "(4 + 2) * x"]:
        result = runner.invoke(cli, ["simplify", problem, "--max-steps=3"])
        assert result.exit_code == 0


def test_cli_simplify_swarm():
    runner = CliRunner()
    result = runner.invoke(cli, ["simplify", "4x + 2x", "--swarm"])
    assert result.exit_code == 0


@pytest.mark.parametrize("agent", ["a3c", "zero"])
def test_cli_train(agent: str):
    runner = CliRunner()
    model_folder = tempfile.mkdtemp()
    result = runner.invoke(
        cli,
        [
            "train",
            agent,
            "poly-like-terms-haystack,poly-grouping",
            model_folder,
            "--verbose",
            "--mcts-sims=3",
            "--episodes=2",
            "--self-play-problems=1",
            "--training-iterations=1",
            "--workers=1",
        ],
    )
    assert result.exit_code == 0

    # Comment this out to keep your model
    shutil.rmtree(model_folder)
