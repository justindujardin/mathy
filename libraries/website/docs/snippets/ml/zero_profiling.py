#!pip install gym
import os
import shutil
import tempfile

from mathy.agents.zero import SelfPlayConfig, self_play_runner
from mathy.cli import setup_tf_env

model_folder = tempfile.mkdtemp()
args = SelfPlayConfig(
    num_workers=1,
    profile=True,
    model_dir=model_folder,
    # All options below here can be deleted if you're actually training
    max_eps=1,
    self_play_problems=1,
    training_iterations=1,
    epochs=1,
    mcts_sims=3,
)

self_play_runner(args)


assert os.path.isfile(os.path.join(args.model_dir, "worker_0.profile"))

# Comment this out to keep your model
shutil.rmtree(model_folder)
