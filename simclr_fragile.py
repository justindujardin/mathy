import os
from typing import Dict, List, Tuple

import gym
import tensorflow as tf
import tqdm
from wasabi import msg

from mathy.agents.contrastive import ContrastiveModelTrainer
from mathy.agents.fragile import SwarmConfig, swarm_solve
from mathy.envs import PolySimplify
from mathy.envs.gym import MathyGymEnv


def get_one_problem(
    env_type: str = "poly", env_difficulty: str = "easy"
) -> Tuple[str, int]:
    """Get one problem text/max_steps tuple"""
    env_name = f"mathy-{env_type}-{env_difficulty}-v0"
    env: MathyGymEnv = gym.make(env_name)
    state, problem = env.mathy.get_initial_state(
        env.env_problem_args, print_problem=False
    )
    return problem.text, state.max_moves


if __name__ == "__main__":
    step = 0
    swarm_runs_per_step = 8
    model_file = "training/contrastive/model"
    env: MathyGymEnv = gym.make("mathy-poly-easy-v0")
    log_dir = os.path.join(os.path.dirname(model_file), "tensorboard")
    writer: tf.summary.SummaryWriter = tf.summary.create_file_writer(log_dir)
    trainer = ContrastiveModelTrainer(input_shape=(512,), writer=writer)
    config = SwarmConfig(
        use_mp=True, history=True, history_names=["observs", "next_observs"]
    )
    if os.path.exists(model_file):
        print(f"Loading checkpoint: {model_file}")
        trainer.model = tf.keras.models.load_model(model_file)
    while True:
        step += 1
        print(f"Step: {step}...")
        compare_problems = []
        compare_steps = []
        contrast_problems = []
        contrast_steps = []
        compare_swarm = None
        contrast_swarm = None
        for i in range(swarm_runs_per_step):
            compare_problem, compare_max_steps = get_one_problem("poly")
            compare_problems.append(compare_problem)
            compare_steps.append(compare_max_steps)

            contrast_problem, contrast_max_steps = get_one_problem("binomial")
            contrast_problems.append(contrast_problem)
            contrast_steps.append(contrast_max_steps)
        # compare_problem, compare_max_steps = ("4x + 2x", 10)
        # contrast_problem, contrast_max_steps = ("14x^2 * x", 10)
        try:
            print(f"    : gathering {swarm_runs_per_step} compare episodes...")
            compare_swarm = swarm_solve(
                compare_problems, config, compare_steps, silent=True
            )
            print(f"    : gathering {swarm_runs_per_step} contrast episodes...")
            contrast_swarm = swarm_solve(
                contrast_problems, config, contrast_steps, silent=True
            )
            trainer.train(
                compare=compare_swarm,
                contrast=contrast_swarm,
                batch_size=swarm_runs_per_step,
            )
            compare_swarm.close()
            contrast_swarm.close()
        except KeyboardInterrupt:
            print("Stopping")
            if compare_swarm is not None:
                compare_swarm.close()
            if contrast_swarm is not None:
                contrast_swarm.close()

        trainer.model.save(model_file)
