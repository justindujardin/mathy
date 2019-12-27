from mathy.cli import setup_tf_env
from mathy.agents.zero import SelfPlayConfig, self_play_runner

import shutil
import tempfile

model_folder = tempfile.mkdtemp()
setup_tf_env()
self_play_cfg = SelfPlayConfig(
    max_eps=1,
    self_play_problems=2,
    verbose=True,
    train=True,
    difficulty="easy",
    topics=["poly", "binomial"],
    lstm_units=16,
    units=32,
    embedding_units=16,
    mcts_sims=5,
    model_dir=model_folder,
    num_workers=2,
    print_training=True,
    training_iterations=1,
)

self_play_runner(self_play_cfg)
# Comment this out to keep your model
shutil.rmtree(model_folder)
