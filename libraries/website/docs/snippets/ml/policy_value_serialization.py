#!pip install gym
import shutil
import tempfile

from mathy.agents.a3c import A3CAgent, AgentConfig
from mathy.agents.model import AgentModel, get_or_create_policy_model
from mathy.cli import setup_tf_env
from mathy.envs import PolySimplify

model_folder = tempfile.mkdtemp()
setup_tf_env()

args = AgentConfig(
    max_eps=3,
    verbose=True,
    topics=["poly"],
    model_dir=model_folder,
    update_gradients_every=4,
    num_workers=1,
    units=4,
    embedding_units=4,
    lstm_units=4,
    print_training=True,
)
instance = A3CAgent(args)
instance.train()
# Load the model back in
model_two = get_or_create_policy_model(
    args=args, predictions=PolySimplify().action_size, is_main=True
)
# Comment this out to keep your model
shutil.rmtree(model_folder)
