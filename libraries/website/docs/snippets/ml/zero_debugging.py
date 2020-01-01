#!pip install gym
from mathy.cli import setup_tf_env
from mathy.agents.zero import SelfPlayConfig, self_play_runner

import shutil
import tempfile

model_folder = tempfile.mkdtemp()
setup_tf_env()
self_play_cfg = SelfPlayConfig(
    # Setting to 1 worker uses single-threaded implementation
    num_workers=1,
    mcts_sims=3,
    max_eps=1,
    self_play_problems=1,
    training_iterations=1,
    verbose=True,
    topics=["poly-combine"],
    model_dir=model_folder,
    print_training=True,
)

self_play_runner(self_play_cfg)
# Comment this out to keep your model
shutil.rmtree(model_folder)
