import shutil
import tempfile
from pathlib import Path

import pytest
from mathy.agent import A3CAgent, AgentConfig
from mathy.api import Mathy, MathyAPIModelState
from mathy.cli import setup_tf_env
from mathy.models import get_model_meta, load_model, load_model_from_init_py, package


def test_models_package_errors() -> None:
    with pytest.raises(SystemExit):
        package("model", "/fake/path", "/fake/pout")
    input_folder = Path(__file__).parent / "test_model_sm"
    output_folder = tempfile.mkdtemp()
    with pytest.raises(SystemExit):
        package("model", input_folder, output_folder, meta_path="fake")
    shutil.rmtree(output_folder)


def test_models_load_model_errors() -> None:
    with pytest.raises(ValueError):
        mt: Mathy = load_model(None)


def test_models_load_model_from_init_py_errors() -> None:
    with pytest.raises(ValueError):
        load_model_from_init_py("./fake/__init__.py")


def test_models_get_model_meta() -> None:
    with pytest.raises(ValueError):
        get_model_meta("./fake/")


def test_models_from_package() -> None:
    setup_tf_env()
    mt: Mathy = load_model("mathy_alpha_sm")
    assert mt is not None
    assert isinstance(mt.state, MathyAPIModelState)
    assert mt.state.model is not None
    assert mt.state.config is not None


def test_models_from_path() -> None:
    setup_tf_env()
    input_folder = Path(__file__).parent / "test_model_sm"
    mt: Mathy = load_model(input_folder)
    assert isinstance(mt.state, MathyAPIModelState)
    assert mt.state.model is not None
    assert mt.state.config is not None


def test_models_package() -> None:
    setup_tf_env()
    input_folder = Path(__file__).parent / "test_model_sm"
    output_folder = tempfile.mkdtemp()
    out_dir = package(
        model_name="zote_the_mighty", input_dir=input_folder, output_dir=output_folder,
    )
    mt: Mathy = load_model(out_dir)
    assert isinstance(mt.state, MathyAPIModelState)
    assert mt.state.model is not None
    assert mt.state.config is not None
    shutil.rmtree(output_folder)


def test_models_train_and_package() -> None:
    setup_tf_env()
    input_folder = tempfile.mkdtemp()
    output_folder = tempfile.mkdtemp()
    setup_tf_env()
    args = AgentConfig(
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
    assert isinstance(mt.state, MathyAPIModelState)
    assert mt.state.model is not None
    assert mt.state.config is not None
    shutil.rmtree(output_folder)
