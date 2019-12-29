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
    train=True,
    difficulty="easy",
    action_strategy="a3c",
    topics=["poly-combine"],
    lstm_units=16,
    units=32,
    embedding_units=24,
    mcts_sims=100,
    model_dir=model_folder,
    num_workers=1,
    profile=False,
    print_training=False,
)
instance = A3CAgent(args)
instance.train()

# Load the model back in
model_two = get_or_create_policy_model(
    args=args, env_actions=PolySimplify().action_size, is_main=True
)

# Comment this out to keep your model
shutil.rmtree(model_folder)
