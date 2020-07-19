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


@pytest.mark.parametrize("use_mp", [True, False])
def test_cli_simplify_swarm(use_mp: bool):
    runner = CliRunner()
    args = ["simplify", "4x + 2x", "--swarm"]
    if use_mp:
        args.append("--parallel")
    result = runner.invoke(cli, args)
    assert result.exit_code == 0


def test_cli_train():
    runner = CliRunner()
    model_folder = tempfile.mkdtemp()
    result = runner.invoke(
        cli,
        [
            "train",
            "poly-like-terms-haystack,poly-grouping",
            model_folder,
            "--verbose",
            "--episodes=2",
            "--workers=1",
        ],
    )
    assert result.exit_code == 0

    # Comment this out to keep your model
    shutil.rmtree(model_folder)
