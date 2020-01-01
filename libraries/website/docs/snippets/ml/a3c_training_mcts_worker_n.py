from mathy.cli import setup_tf_env
from mathy.agents.a3c import A3CAgent, A3CConfig
import shutil
import tempfile

model_folder = tempfile.mkdtemp()
setup_tf_env()

args = A3CConfig(
    action_strategy="mcts_worker_n",
    max_eps=1,
    verbose=True,
    topics=["poly-combine"],
    model_dir=model_folder,
    num_workers=2,
    clip_grouping_control=False,
    print_training=True,
)
A3CAgent(args).train()
# Comment this out to keep your model
shutil.rmtree(model_folder)
