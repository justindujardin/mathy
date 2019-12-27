import shutil
import tempfile

from mathy.agents.a3c import A3CConfig
from mathy.agents.policy_value_model import PolicyValueModel, get_or_create_policy_model
from mathy.cli import setup_tf_env
from mathy.envs import PolySimplify

model_folder = tempfile.mkdtemp()
setup_tf_env()
args = A3CConfig(model_dir=model_folder, verbose=True)
model: PolicyValueModel = get_or_create_policy_model(
    args=args, env_actions=PolySimplify().action_size,
)
model.save()
# Comment this out to keep your model
shutil.rmtree(model_folder)
