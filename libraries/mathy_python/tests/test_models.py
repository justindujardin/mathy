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


def test_models_package():
    setup_tf_env()
    input_folder = Path(__file__).parent / "test_model_sm"
    output_folder = tempfile.mkdtemp()
    out_folder = package(
        model_name="cool_model", input_dir=input_folder, output_dir=output_folder,
    )
    mt: Mathy = load_model(out_folder)
    assert mt is not None
    assert mt.model is not None
    assert mt.config is not None
    shutil.rmtree(output_folder)
