# coding: utf8
import json
import os
import random
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import numpy
import plac
import tensorflow as tf
from colr import color
from mathy.agent.controller import MathModel
from mathy.agent.curriculum.level1 import (
    combine_forced,
    lesson_plan,
    lesson_plan_2,
    lesson_plan_3,
    lesson_quick,
    moves_per_complexity,
    white_belt,
    yellow_belt,
    green_belt,
    green_belt_practice,
    purple_belt_practice,
    white_belt_practice,
    node_control,
)
from mathy.agent.training.actor_mcts import ActorMCTS
from mathy.agent.training.lessons import LessonExercise, LessonPlan, build_lesson_plan
from mathy.agent.training.math_experience import (
    MathExperience,
    balanced_reward_experience_samples,
)
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
from mathy.math_game import MathGame

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


lessons = {
    "node_control": node_control,
    "practice1": white_belt_practice,
    "exam2": yellow_belt,
    "practice3": green_belt_practice,
    "purple_belt_practice": purple_belt_practice,
    "exam3": green_belt,
    "old_lesson3": lesson_plan_3,
    "old_lesson1": lesson_plan,
    "white_belt": white_belt,
    "yellow_belt": yellow_belt,
    "green_belt": green_belt,
    "green_belt_practice": green_belt_practice,
    "dev": lesson_quick,
}


@plac.annotations(
    model_dir=(
        "The name of the model to examine the performance of.",
        "positional",
        None,
        str,
    ),
    lesson_id=("The lesson plan to execute by ID", "option", "l", str),
    mcts_sims=("The number of MCTS rollouts to perform per move", "option", "s", int),
    num_exploration_moves=(
        "The number of exploration moves to perform before switching to exploitation",
        "option",
        "exploration",
        int,
    ),
    epsilon=("The epsilon value for MCTS search", "option", "epsilon", float),
)
def main(
    model_dir,
    lesson_id="green_belt",
    mcts_sims=500,
    num_exploration_moves=0,
    epsilon=0.0,
):
    global lessons
    controller = MathGame(verbose=True)
    mathy = MathModel(controller.action_size, model_dir)
    experience = MathExperience(mathy.model_dir, 128)
    mathy.start()
    if lesson_id not in lessons:
        raise ValueError(
            f"[exam] ERROR: '{lesson_id}' not found in ids. Valid lessons are: {', '.join(lessons)} "
        )
    plan = lessons[lesson_id]
    plan_lessons = plan.lessons
    num_solved = 0
    num_failed = 0
    model = mathy
    print("[exam] lesson order: {}".format([l.name for l in plan_lessons]))
    # we fill this with episode rewards and when it's a fixed size we
    # dump the average value to tensorboard
    ep_reward_buffer = []
    while len(plan_lessons) > 0:
        lesson = plan_lessons.pop(0)
        controller.lesson = lesson
        print("\n[exam] {} - {}...".format(plan.name.lower(), lesson.name.lower()))
        # Fill up a certain amount of experience per problem type
        lesson_experience_count = 0
        if lesson.num_observations is not None:
            iter_experience = lesson.num_observations
        else:
            iter_experience = short_term_size
        while lesson_experience_count < iter_experience:
            env_state, complexity = controller.get_initial_state(print_problem=False)
            complexity_value = complexity * moves_per_complexity
            controller.max_moves = (
                lesson.max_turns if lesson.max_turns is not None else complexity_value
            )
            # generate a new problem now that we've set the max_turns
            env_state, complexity = controller.get_initial_state()
            mcts = MCTS(controller, model, epsilon, mcts_sims)
            actor = ActorMCTS(mcts, num_exploration_moves)
            final_result = None
            time_steps = []
            episode_steps = 0
            start = time.time()
            while final_result is None:
                episode_steps = episode_steps + 1
                env_state, train_example, final_result = actor.step(
                    controller, env_state, model, time_steps
                )

            elapsed = time.time() - start
            episode_examples, episode_reward, is_win = final_result
            lesson_experience_count += len(episode_examples)
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
                        lesson.name.upper(),
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
        ep_reward_buffer = []

    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
