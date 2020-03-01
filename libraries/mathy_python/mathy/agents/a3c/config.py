from ...agents.base_config import BaseConfig


class A3CConfig(BaseConfig):
    # Update frequencey for the Worker to sync with the Main model.
    #
    # Indicates the maximum number of steps to take in an episode before
    # syncing the replay buffer and gradients.
    update_gradients_every: int = 8

    normalization_style: str = "layer"

    # Strategy for introducing MCTS into the A3C agent training process
    #
    #   - "a3c" Do not use MCTS when training the A3C agent
    #   - "mcts_worker_0" uses MCTS for observations on the greediest worker. This
    #                usually looks the best visually, because mcts_worker_0 is print
    #                to stdout, so it results in lots of green. It's unclear that
    #                it helps improve the A3C agent performance after MCTS is removed.
    #   - "mcts_worker_n" uses MCTS for observations on all workers except worker_0.
    #                This adds the strength of MCTS to observation gathering, without
    #                biasing the observed strength of the model (because only worker_0)
    #                reports statistics.
    action_strategy = "a3c"
    # MCTS provides higher quality observations at extra computational cost.
    mcts_sims: int = 200

    main_worker_use_epsilon = False
    e_greedy_min = 0.01
    e_greedy_max = 0.1
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
    td_lambda: float = 0.2

    # The "Teacher" will start evaluating after this many initial episodes
    teacher_start_evaluations_at_episode = 50
    # The "Teacher" evaluates the win/loss record of the agent every (n) episodes
    teacher_evaluation_steps = 20
    # If the agent wins >= this value, promote to the next difficulty class
    # Wild-ass guess inspired by:
    # https://uanews.arizona.edu/story/learning-optimized-when-we-fail-15-time
    # If 85 is optimal, when you go beyond 85 + buffer it's time to move up... |x_X|
    teacher_promote_wins = 0.95
    # If the agent loses >= this value, demot to the previous difficulty class
    teacher_demote_wins = 0.50

    # When profile is true, each A3C worker thread will output a .profile
    # file in the model save path when it exits.
    profile: bool = False

    # Verbose setting to print out worker_0 training steps. Useful for trying
    # to find problems.
    print_training: bool = False

    # Print mode for output. "terminal" is the default
    print_mode: str = "terminal"
