# coding: utf8
"""Train a mathy model on a given input set, then run a lesson evaluation and exit"""
import json
import os
import random
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from shutil import copyfile

import numpy
import plac
import tensorflow as tf
from colr import color

from mathy.agent.controller import MathModel
from mathy.agent.curriculum.level1 import combine_like_terms_complexity_challenge
from mathy.agent.training.actor_mcts import ActorMCTS
from mathy.agent.training.lessons import LessonExercise, LessonPlan, build_lesson_plan
from mathy.agent.training.math_experience import MathExperience
from mathy.agent.training.mcts import MCTS
from mathy.agent.training.practice_runner import (
    ParallelPracticeRunner,
    PracticeRunner,
    RunnerConfig,
)
from mathy.agent.training.practice_session import PracticeSession
from mathy.agent.training.problems import (
    MODE_SIMPLIFY_POLYNOMIAL,
    get_rand_vars,
    maybe_int,
    rand_var,
    simplify_multiple_terms,
)
from mathy.core.parser import ExpressionParser, ParserException
from mathy.environment_state import INPUT_EXAMPLES_FILE_NAME
from mathy.math_game import MathGame

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")

moves_per_complexity = 3

lesson_plan = build_lesson_plan(
    "combine_like_terms_1",
    [
        LessonExercise(
            lesson_name="three_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
        ),
        LessonExercise(
            lesson_name="needle_in_haystack",
            problem_count=32,
            problem_fn=lambda: combine_like_terms_complexity_challenge(),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=3,
            mcts_sims=100,
        ),
        LessonExercise(
            lesson_name="four_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
        ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
        ),
        LessonExercise(
            lesson_name="seven_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=100,
        ),
    ],
)
hard_plan = build_lesson_plan(
    "train_evaluation_hard",
    [
        LessonExercise(
            lesson_name="15_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(15),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
        ),
        LessonExercise(
            lesson_name="24_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(24),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
        ),
    ],
)


@plac.annotations(
    model_dir=(
        "The name of the model to train. This changes the output folder.",
        "positional",
        None,
        str,
    ),
    examples_file=(
        "The location of a JSONL file with observations to train the model on",
        "positional",
        None,
        str,
    ),
    transfer_from=(
        "The name of another model to warm start this one from. Think Transfer Learning",
        "positional",
        None,
        str,
    ),
    no_train=(
        "When set do not train the network before evaluation. Useful for evaluating existing models",
        "flag",
        "n",
    ),
)
def main(model_dir, examples_file, transfer_from=None, no_train=False):

    plan = lesson_plan
    lessons = plan.lessons[:]
    num_solved = 0
    num_failed = 0
    num_rollouts = 100
    num_exploration_moves = 0
    epsilon = 0
    initial_train_iterations = 6
    controller = MathGame(verbose=True)
    input_examples = Path(examples_file)
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        print("Making model_dir: {}".format(model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)
    if input_examples.is_file():
        print("Copying examples into place: {}".format(model_dir))
        train_dir = model_dir / "train"
        if not train_dir.is_dir():
            train_dir.mkdir(parents=True, exist_ok=True)
        copyfile(str(input_examples), model_dir / "train" / INPUT_EXAMPLES_FILE_NAME)

    mathy = MathModel(
        controller.action_size,
        model_dir,
        init_model_dir=transfer_from,
        init_model_overwrite=True,
        learning_rate=0.01,
    )
    experience = MathExperience(mathy.model_dir)
    mathy.start()

    if no_train:
        print(
            color(
                "Skipping initial training because no_train flag is set", fore="yellow"
            )
        )
    else:
        print(
            color(
                "Training for {} iterations on existing knowledge before beginning class".format(
                    initial_train_iterations
                ),
                fore="blue",
            )
        )
        mathy.epochs = initial_train_iterations
        mathy.train(experience.short_term, experience.long_term, train_all=True)
        # mathy.train(experience.short_term, experience.long_term)

    print(color("Evaluting model performance on exam questions!", fore="green"))
    ep_reward_buffer = []
    while len(lessons) > 0:
        lesson = lessons.pop(0)
        controller.lesson = lesson
        print("\n{} - {}...".format(plan.name.upper(), lesson.name.upper()))
        for i in range(lesson.problem_count):
            env_state, complexity = controller.get_initial_state(print_problem=False)
            complexity_value = complexity * moves_per_complexity
            controller.max_moves = (
                lesson.max_turns if lesson.max_turns is not None else complexity_value
            )
            # generate a new problem now that we've set the max_turns
            env_state, complexity = controller.get_initial_state()
            mcts = MCTS(controller, mathy, epsilon, num_rollouts)
            actor = ActorMCTS(mcts, num_exploration_moves)
            final_result = None
            time_steps = []
            episode_steps = 0
            start = time.time()
            while final_result is None:
                episode_steps = episode_steps + 1
                env_state, train_example, final_result = actor.step(
                    controller, env_state, mathy, time_steps
                )

            elapsed = time.time() - start
            episode_examples, episode_reward, is_win = final_result
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
                    " -- duration({}) outcome({})".format(
                        str(timedelta(seconds=elapsed)), outcome
                    ),
                    fore=fore,
                    style="bright",
                )
            )
            experience.add_batch(episode_examples)
            ep_reward_buffer.append(episode_reward / episode_steps)
        # Print result rewards
        var_name = "{}/step_avg_reward_{}".format(
            plan.name.replace(" ", "_").lower(), lesson.name.replace(" ", "_").lower()
        )
        var_data = numpy.mean(ep_reward_buffer) if len(ep_reward_buffer) > 0 else 0.0
        print(color("[{} = {}]".format(var_name, var_data), fore="magenta"))
        ep_reward_buffer = []

    print(
        color(
            "\n\n=== Evaluation complete solve({}) fail({}) ===\n\n".format(
                num_solved, num_failed
            ),
            fore="blue",
            style="bright",
        )
    )
    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
