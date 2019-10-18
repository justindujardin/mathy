from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class A3CArgs(BaseModel):
    units: int
    embedding_units: int
    lstm_units: int
    topics: List[str] = ["poly"]
    difficulty: Optional[str] = None
    model_dir: str = "/tmp/a3c-training/"
    model_name: str = "model.h5"
    init_model_from: Optional[str] = None
    train: bool = False
    verbose: bool = False
    # The history size for the greedy worker
    greedy_history_size: int = 1024
    # History size for exploratory workers
    history_size: int = 512
    # Size at which it's okay to start sampling from the memory
    ready_at: int = 256
    lr: float = 3e-4
    update_freq: int = 25
    max_eps: int = 15000
    gamma: float = 0.99

    # Strategy for introducing MCTS into the A3C agent training process
    #
    #   - "a3c" Do not use MCTS when training the A3C agent
    #   - "mcts" Use MCTS for everything. The slowest option, generates the best
    #            samples.
    #   - "mcts_worker_0" uses MCTS for observations on the greediest worker. This
    #                usually looks the best visually, because mcts_worker_0 is print
    #                to stdout, so it results in lots of green. It's unclear that
    #                it helps improve the A3C agent performance after MCTS is removed.
    #   - "mcts_worker_n" uses MCTS for observations on all workers except worker_0.
    #                This adds the strength of MCTS to observation gathering, without
    #                biasing the observed strength of the model (because only worker_0)
    #                reports statistics.
    #   - "mcts_e_unreal" each worker uses an eGreedy style epsilon check at the
    #                beginning of each episode. If the value is less than epsilon the
    #                episode steps will be enhanced using MCTS. The enhanced examples
    #                are stored to the Experience replace for UNREAL training. This
    #                causes the buffer to fill more slowly, but the examples in it
    #                are guaranteed to be high quality compared to a normal agent.
    #                I hypothesize that filling the buffer with fewer higher-quality
    #                examples will have the same effect as selecting prioritized
    #                experiences for replay (as in Ape-X, R2D2, etc)
    #   - "mcts_recover" An average "steps to solve" for each problem type/difficulty is tracked
    #                and when the agent exceeds that step number in a problem, MCTS is applied
    #                for the remaining steps. This is an attempt to convert near miss (all negative)
    #                episodes into "weak win" ones. The idea is that agents struggle to overcome
    #                the sign-flipping effect of episode loss/wins
    action_strategy = "a3c"
    # MCTS provides higher quality observations at extra computational cost.
    mcts_sims: int = 10
    mcts_recover_time_threshold: float = 0.66
    # When using "" action strategy, this is the epsilon that will trigger an MCTS
    # episode when random is less than it.
    unreal_mcts_epsilon: float = 0.05

    # NOTE: scaling down h_loss is observed to be important to keep it from
    #       destabilizing the overall loss when it grows very small
    entropy_loss_scaling = 0.1

    # How much to scale down loss values from auxiliary tasks
    aux_tasks_weight_scale = 0.1

    main_worker_use_epsilon = False
    e_greedy_min = 0.01
    e_greedy_max = 0.4
    # Worker's sleep this long between steps to allow
    # other threads time to process. This is useful for
    # running more threads than you have processors to
    # get a better diversity of experience.
    worker_wait: float = 0.5

    # The number of worker agents to create.
    num_workers: int = 3

    # The "Teacher" evaluates the win/loss record of the agent every (n) episodes
    teacher_evaluation_steps = 25
    # If the agent wins >= this value, promote to the next difficulty class
    teacher_promote_wins = 0.75
    # If the agent loses >= this value, demot to the previous difficulty class
    teacher_demote_wins = 0.3

    # When profile is true, each A3C worker thread will output a .profile
    # file in the model save path when it exits.
    profile: bool = False

    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False

    # When training on the experience replay buffer, burn-in the stored RNN states 
    # against the current model for (n) steps before processing the replay examples
    unreal_burn_in_steps: int = 1

    # Whether to use the reward prediction aux task
    use_reward_prediction = True

    # Whether to use the value replay aux task
    use_value_replay = True

    # Whether to use the grouping change aux task
    use_grouping_control = True
