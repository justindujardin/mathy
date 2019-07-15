import os
import time
from datetime import timedelta
from typing import List, Type

import numpy
import plac
import tensorflow as tf
from colr import color

from mathy.agent.controller import MathModel
from mathy.agent.training.actor_mcts import ActorMCTS
from mathy.agent.training.math_experience import (
    MathExperience,
    balanced_reward_experience_samples,
)
from mathy.agent.training.mcts import MCTS
from mathy.envs import (
    MathyBinomialDistributionEnv,
    MathyComplexTermSimplificationEnv,
    MathyPolynomialSimplificationEnv,
)
from mathy.mathy_env import mathy_core_rules
from mathy.mathy_env import MathyEnv
from mathy.types import MathyEnvObservation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


@plac.annotations(
    env=("Environment to load", "positional", None, str),
    model_dir=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    ),
    transfer_from=(
        "Transfer weights from another model by its folder path",
        "positional",
        None,
        str,
    ),
    difficulty=(
        "The arbitrary integer difficulty of problems to generate",
        "option",
        "d",
        int,
    ),
    learning_rate=("The learning rate to use when training", "option", "lr", float),
    turns_per_complexity=(
        "The number of moves to allocate per unit of problem complexity",
        "option",
        "t",
        int,
    ),
    verbose=(
        "When true, print all problem moves rather than just during evaluation",
        "flag",
        "v",
    ),
)
def main(
    env: str,
    model_dir: str,
    transfer_from=None,
    verbose=False,
    turns_per_complexity=4,
    difficulty=3,
    # Learning rate found via some hyperparam exploration.
    learning_rate=2e-4,
):

    environments = {
        "poly": [MathyPolynomialSimplificationEnv],
        "binomial": [MathyBinomialDistributionEnv],
        "complex": [MathyComplexTermSimplificationEnv],
        "mixed": [
            MathyPolynomialSimplificationEnv,
            MathyComplexTermSimplificationEnv,
            MathyBinomialDistributionEnv,
        ],
    }
    if env not in environments:
        raise EnvironmentError(f"Invalid env, must be one of: {environments.keys()}")
    # How many observations to gather between training sessions.
    iter_experience = 128
    min_train_experience = 128
    eval_interval = 2
    short_term_size = 2048
    long_term_size = 8192 * 3
    counter = 0
    training_epochs = 4
    action_size = len(mathy_core_rules())
    mathy = MathModel(
        action_size,
        model_dir,
        init_model_dir=transfer_from,
        learning_rate=learning_rate,
        long_term_size=long_term_size,
        epochs=training_epochs,
    )
    experience = MathExperience(mathy.model_dir, short_term_size)
    mathy.start()
    while True:
        print(f"Iteration: {counter}")
        print(f"Moves/complexity: {turns_per_complexity}")
        print(f"Difficulty: {difficulty}")
        counter = counter + 1
        eval_run = (
            bool(counter % eval_interval == 0)
            and experience.count >= min_train_experience
        )
        num_solved = 0
        num_failed = 0

        if eval_run:
            print("\n\n=== Evaluating model with exploitation strategy ===")
            mathy.stop()
            mathy_eval = MathModel(
                action_size,
                model_dir,
                init_model_dir=os.path.abspath(mathy.model_dir),
                # We want to initialize from the training model for each evaluation. (?)
                init_model_overwrite=True,
                is_eval_model=True,
                learning_rate=learning_rate,
                epochs=training_epochs,
                long_term_size=long_term_size,
            )
            mathy_eval.start()
        model = mathy_eval if eval_run else mathy
        # we fill this with episode rewards and when it's a fixed size we
        # dump the average value to tensorboard
        ep_reward_buffer: List[float] = []
        # Fill up a certain amount of experience per problem type
        lesson_experience_count = 0
        lesson_problem_count = 0
        while lesson_experience_count < iter_experience:
            env_classes: List[Type[MathyEnv]] = environments[env]  # type: ignore
            env_class = env_classes[0]
            for i, clazz in enumerate(reversed(env_classes)):
                if lesson_problem_count % (len(env_classes) - i) == 0:
                    env_class = clazz
                    break
            mathy_env = env_class(verbose=True)
            env_name = str(mathy_env.__class__.__name__)
            mathy_env.verbose = eval_run or verbose
            print(f"{env_name}")
            # generate a new problem
            options = {
                "difficulty": difficulty,
                "turns_per_complexity": turns_per_complexity,
            }
            env_state, prob = mathy_env.get_initial_state(options)

            # Configure MCTS options for train/eval
            if eval_run:
                num_rollouts = 500
                num_exploration_moves = 0
                epsilon = 0.0
            else:
                num_rollouts = 250
                num_exploration_moves = int(mathy_env.max_moves * 0.8)
                epsilon = 0.9

            # Execute episode
            model = mathy_eval if eval_run else mathy
            mcts = MCTS(mathy_env, model, epsilon, num_rollouts)
            actor = ActorMCTS(mcts, num_exploration_moves)
            final_result = None
            time_steps: List[MathyEnvObservation] = []
            episode_steps = 0
            start = time.time()
            while final_result is None:
                episode_steps = episode_steps + 1
                env_state, train_example, final_result = actor.step(
                    mathy_env, env_state, model, time_steps
                )

            elapsed = time.time() - start
            episode_examples, episode_reward, is_win = final_result
            lesson_experience_count += len(episode_examples)
            lesson_problem_count += 1
            if is_win:
                num_solved = num_solved + 1
                outcome = "solved"
                fore = "green"
            else:
                num_failed = num_failed + 1
                outcome = "failed"
                fore = "red"
            print(
                color(
                    "{} [{}/{}] -- duration({}) outcome({})".format(
                        env_name,
                        lesson_experience_count,
                        iter_experience,
                        str(timedelta(seconds=elapsed)),
                        outcome,
                    ),
                    fore=fore,
                    style="bright",
                )
            )
            experience.add_batch(episode_examples)
            ep_reward_buffer.append(episode_reward / episode_steps)

        # Train if we have enough data
        if experience.count > min_train_experience:
            model.train(
                experience.short_term,
                experience.long_term,
                sampling_fn=balanced_reward_experience_samples,
            )
        else:
            print(
                color(
                    "Need {} observations for training but have {}".format(
                        min_train_experience, experience.count
                    ),
                    fore="yellow",
                    style="bright",
                )
            )
            continue

        summary_writer = tf.summary.create_file_writer(model.model_dir)
        with summary_writer.as_default():
            global_step = model.network.get_variable_value("global_step")
            var_name = "{}/step_avg_reward".format(env_name.replace(" ", "_").lower())
            var_data = (
                numpy.mean(ep_reward_buffer) if len(ep_reward_buffer) > 0 else 0.0
            )
            print(
                color(
                    "{} [{} = {}]".format(global_step, var_name, var_data),
                    fore="magenta",
                )
            )
            tf.summary.scalar(name=var_name, data=var_data, step=global_step)

        summary_writer.close()
        ep_reward_buffer = []

        if eval_run:
            print(
                color(
                    "\n\n=== Evaluation complete solve({}) fail({}) ===\n\n".format(
                        num_solved, num_failed
                    ),
                    fore="blue",
                    style="bright",
                )
            )
            mathy_eval.stop()
            mathy.start()

    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
