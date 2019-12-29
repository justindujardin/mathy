from mathy.cli import setup_tf_env

setup_tf_env(use_mp=True)

from mathy.agents.zero import SelfPlayConfig, self_play_runner

import shutil
import tempfile

model_folder = tempfile.mkdtemp()
self_play_cfg = SelfPlayConfig(
    # This option is set to allow the script to run quickly.
    # You'll probably want a much larger value during training. 10000?
    max_eps=1,
    # This is set to allow training after the first set of problems.
    # You'll probalby want this to be more like 128
    batch_size=1,
    # This is set to only do 2 problems before training. You guessed
    # it, in order to keep things snappy. Try 100.
    self_play_problems=2,
    # This is set to 1 in order to exit after the first gather/training loop.
    training_iterations=1,
    # This is normally larger, try 10
    epochs=1,
    # This is a tiny model, designed to be fast for testing.
    lstm_units=16,
    units=32,
    embedding_units=16,
    # The number of MCTS sims directly correlates with finding quality
    # actions. Normally I would set this to something like 100, 250, 500
    # depending on the problem difficulty.
    mcts_sims=2,
    # This can be scaled to however many CPUs you have available. Going
    # higher than your CPU count does not produce good performance usually.
    num_workers=2,
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
