from mathy.cli import setup_tf_env
from mathy.agents.zero import SelfPlayConfig, self_play_runner

import shutil
import tempfile

model_folder = tempfile.mkdtemp()
setup_tf_env()
self_play_cfg = SelfPlayConfig(
    # Setting to 1 worker uses single-threaded implementation
    num_workers=1,
    max_eps=1,
    self_play_problems=1,
    verbose=True,
    train=True,
    difficulty="easy",
    topics=["poly-combine"],
    lstm_units=16,
    units=32,
    embedding_units=16,
    mcts_sims=1,
    model_dir=model_folder,
    print_training=True,
    training_iterations=1,
)

self_play_runner(self_play_cfg)
# Comment this out to keep your model
shutil.rmtree(model_folder)
