from typing import List, Optional

from pydantic import BaseModel
from .. import about


class AgentConfig(BaseModel):
    def __hash__(self) -> int:
        return hash(",".join([f"{k}{v}" for k, v in self.dict().items()]))

    class Config:
        extra = "allow"

    name: str = "unnamed_model"
    # The version of the model. You probably don't want to change this.
    version: str = "0.0.1"
    description: str = "Mathy.ai trained model"
    license: str = "CC BY-SA 3.0"
    author: str = about.__author__
    email: str = about.__email__
    url: str = about.__uri__
    mathy_version: str = f">={about.__version__},<1.0.0"
    units: int = 64
    max_len: int = 128
    embedding_units: int = 32
    lstm_units: int = 128
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = "easy"
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model"
    verbose: bool = False
    # Initial learning rate that decays over time.
    lr: float = 6e-5
    max_eps: int = 150000
    # How often to write histograms to tensorboard (in training steps)
    summary_interval: int = 100
    gamma: float = 0.95
    # The number of worker agents to create.
    num_workers: int = 3
    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False
    # Print mode for output. "terminal" is the default, also supports "attention"
    # NOTE: attention is gone (like... the layer)
    print_mode: str = "terminal"
    # The number of timesteps use when making predictions. This includes the current timestep and
    # (n - 1) previous timesteps
    prediction_window_size: int = 32
    # The lambda value for generalized lambda returns to calculate value loss
    # 0.0 = bootstrap values, 1.0 = discounted
    td_lambda: float = 1.0
    # Update frequency for the Worker to sync with the Main model.
    #
    # Indicates the maximum number of steps to take in an episode before
    # syncing the replay buffer and gradients.
    update_gradients_every: int = 32
    # The number of worker agents to create.
    num_workers: int = 3

    policy_fn_entropy_cost = 0.05
    policy_args_entropy_cost = 0.75

    # The "Teacher" will start evaluating after this many initial episodes
    teacher_start_evaluations_at_episode = 20
    # The "Teacher" evaluates the win/loss record of the agent every (n) episodes
    teacher_evaluation_steps = 3
    # If the agent wins >= this value, promote to the next difficulty class
    # Wild-ass guess inspired by:
    # https://uanews.arizona.edu/story/learning-optimized-when-we-fail-15-time
    # If 85 is optimal, when you go beyond 85 + buffer it's time to move up... |x_X|
    teacher_promote_wins = 0.90
    # If the agent loses >= this value, demote to the previous difficulty class
    teacher_demote_wins = 0.66

    # When profile is true, each A3C worker thread will output a .profile
    # file in the model save path when it exits.
    profile: bool = False

    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False

    # Print mode for output. "terminal" is the default
    print_mode: str = "terminal"
