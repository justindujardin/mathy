import shutil
import tempfile
from unittest.mock import patch

import pytest

from ..mathy.agents.a3c import A3CAgent, A3CConfig
from ..mathy.agents.policy_value_model import (
    PolicyValueModel,
    get_or_create_policy_model,
)
from ..mathy.cli import setup_tf_env
from ..mathy.envs import PolySimplify
from ..mathy.mathy import Mathy
from ..mathy.models import load_model, package
from pathlib import Path


def test_models_from_package():
    setup_tf_env()
    mt: Mathy = load_model("mathy_alpha_sm")
    assert mt is not None
    assert mt.model is not None
    assert mt.config is not None


def test_models_from_path():
    setup_tf_env()
    input_folder = Path(__file__).parent / "test_model_sm"
    mt: Mathy = load_model(input_folder)
    assert mt is not None
    assert mt.model is not None
    assert mt.config is not None


def test_models_package():
    setup_tf_env()
    input_folder = Path(__file__).parent / "test_model_sm"
    output_folder = tempfile.mkdtemp()
    out_dir = package(
        model_name="zote_the_mighty", input_dir=input_folder, output_dir=output_folder,
    )
    mt: Mathy = load_model(out_dir)
    assert mt is not None
    assert mt.model is not None
    assert mt.config is not None
    shutil.rmtree(output_folder)


def test_models_train_and_package():
    setup_tf_env()
    input_folder = tempfile.mkdtemp()
    output_folder = tempfile.mkdtemp()
    setup_tf_env()
    args = A3CConfig(
        max_eps=1,
        topics=["poly-combine"],
        model_dir=input_folder,
        num_workers=1,
        units=3,
        lstm_units=3,
        embedding_units=3,
    )
    agent = A3CAgent(args)
    agent.train()
    out_dir = package(
        model_name="zote_the_mighty", input_dir=input_folder, output_dir=output_folder,
    )
    mt: Mathy = load_model(out_dir)
    assert mt is not None
    assert mt.model is not None
    assert mt.config is not None
    shutil.rmtree(output_folder)
