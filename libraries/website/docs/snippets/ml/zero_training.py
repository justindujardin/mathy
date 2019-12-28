from mathy.cli import setup_tf_env
from mathy.agents.zero import SelfPlayConfig, self_play_runner

import shutil
import tempfile

model_folder = tempfile.mkdtemp()
setup_tf_env()
self_play_cfg = SelfPlayConfig(
    max_eps=1,
    self_play_problems=2,
    batch_size=1,
    num_workers=1,
    epochs=1,
    training_iterations=1,
    mcts_sims=2,
    lstm_units=16,
    units=32,
    embedding_units=16,
    verbose=True,
    train=True,
    difficulty="easy",
    topics=["poly-combine"],
    model_dir=model_folder,
    print_training=True,
)

self_play_runner(self_play_cfg)
# Comment this out to keep your model
shutil.rmtree(model_folder)
