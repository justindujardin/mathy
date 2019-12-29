import shutil
import tempfile

from mathy.agents.a3c import A3CAgent, A3CConfig
from mathy.agents.policy_value_model import PolicyValueModel, get_or_create_policy_model
from mathy.cli import setup_tf_env
from mathy.envs import PolySimplify

model_folder = tempfile.mkdtemp()
setup_tf_env()

args = A3CConfig(
    max_eps=1,
    verbose=True,
    topics=["poly-combine"],
    model_dir=model_folder,
    num_workers=1,
)
instance = A3CAgent(args)
instance.train()

# Load the model back in
model_two = get_or_create_policy_model(
    args=args, env_actions=PolySimplify().action_size, is_main=True
)

# Comment this out to keep your model
shutil.rmtree(model_folder)
