from typing import List, Optional

from pydantic import BaseModel
from .. import about


class AgentConfig(BaseModel):
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
    # One of "batch" or "layer"
    normalization_style = "layer"
    # The number of timesteps use when making predictions. This includes the current timestep and
    # (n - 1) previous timesteps
    prediction_window_size: int = 6
    units: int = 64
    embedding_units: int = 128
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = None
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model"
    verbose: bool = False
    # Initial learning rate that decays over time.
    lr_initial: float = 0.01
    lr_decay_steps: int = 100
    lr_decay_rate: float = 0.96
    lr_decay_staircase: bool = True
    max_eps: int = 15000
    # How often to write histograms to tensorboard (in training steps)
    summary_interval: int = 100
    gamma: float = 0.99
    # The number of worker agents to create.
    num_workers: int = 3
    # The lambda value for generalized lambda returns to calculate value loss
    # 0.0 = bootstrap values, 1.0 = discounted
    td_lambda: float = 0.2
    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False
    # This is very verbose and prints every model.call time
    print_model_call_times: bool = False
    # Print mode for output. "terminal" is the default, also supports "attention"
    # NOTE: attention is gone (like... the layer)
    print_mode: str = "terminal"
    # Update frequencey for the Worker to sync with the Main model.
    #
    # Indicates the maximum number of steps to take in an episode before
    # syncing the replay buffer and gradients.
    update_gradients_every: int = 8
    main_worker_use_epsilon = False
    e_greedy_min = 0.01
    e_greedy_max = 0.04
    # Worker's sleep this long between steps to allow
    # other threads time to process. This is useful for
    # running more threads than you have processors to
    # get a better diversity of experience.
    worker_wait: float = 0.01

    # The number of worker agents to create.
    num_workers: int = 3

    # NOTE: scaling down h_loss is observed to be important to keep it from
    #       destabilizing the overall loss when it grows very small
    entropy_loss_scaling = 0.05
    # Whether to scale entropy loss so it's 0-1
    normalize_entropy_loss = True
    # Scale policy loss down by sequence length to make loss length invariant
    normalize_pi_loss = True

    # How much to scale down loss values from auxiliary tasks
    aux_tasks_weight_scale = 1.0
    # The lambda value for generalized lambda returns to calculate value loss
    # 0.0 = bootstrap values, 1.0 = discounted
    td_lambda: float = 0.5

    # The "Teacher" will start evaluating after this many initial episodes
    teacher_start_evaluations_at_episode = 500
    # The "Teacher" evaluates the win/loss record of the agent every (n) episodes
    teacher_evaluation_steps = 20
    # If the agent wins >= this value, promote to the next difficulty class
    # Wild-ass guess inspired by:
    # https://uanews.arizona.edu/story/learning-optimized-when-we-fail-15-time
    # If 85 is optimal, when you go beyond 85 + buffer it's time to move up... |x_X|
    teacher_promote_wins = 0.90
    # If the agent loses >= this value, demot to the previous difficulty class
    teacher_demote_wins = 0.65

    # When profile is true, each A3C worker thread will output a .profile
    # file in the model save path when it exits.
    profile: bool = False

    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False

    # Print mode for output. "terminal" is the default
    print_mode: str = "terminal"
