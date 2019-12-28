from mathy.cli import setup_tf_env
from mathy.agents.a3c import A3CAgent, A3CConfig
import shutil
import tempfile

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
    num_workers=2,
    profile=False,
    print_training=True,
)
instance = A3CAgent(args)
instance.train()
# Comment this out to keep your model
shutil.rmtree(model_folder)
