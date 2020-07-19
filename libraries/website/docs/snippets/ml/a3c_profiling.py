#!pip install gym
import os
from mathy.cli import setup_tf_env
from mathy.agent import A3CAgent, AgentConfig
import shutil
import tempfile

model_folder = tempfile.mkdtemp()
setup_tf_env()

args = AgentConfig(
    profile=True,
    max_eps=2,
    verbose=True,
    topics=["poly-grouping"],
    model_dir=model_folder,
    num_workers=2,
    print_training=True,
)
A3CAgent(args).train()

assert os.path.isfile(os.path.join(args.model_dir, "worker_0.profile"))
assert os.path.isfile(os.path.join(args.model_dir, "worker_1.profile"))

# Comment this out to keep your model
shutil.rmtree(model_folder)
