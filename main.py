# coding: utf8
"""Pre-train the math vectors to improve agent performance"""
import json
import os
import tempfile
from pathlib import Path
import random
from colr import color
import numpy
import plac
import time
from mathzero.training.lessons import LessonExercise, LessonPlan
from mathzero.core.parser import ExpressionParser, ParserException
from mathzero.embeddings.math_game import MathGame
from mathzero.model.math_model import MathModel
from mathzero.training.lessons import LessonExercise, build_lesson_plan
from mathzero.training.practice_runner import (
    ParallelPracticeRunner,
    PracticeRunner,
    RunnerConfig,
)
from mathzero.training.practice_session import PracticeSession
from mathzero.training.problems import (
    MODE_SIMPLIFY_POLYNOMIAL,
    simplify_multiple_terms,
    rand_var,
    maybe_int,
    get_rand_vars,
)
from mathzero.embeddings.math_experience import MathExperience
from mathzero.training.mcts import MCTS
from mathzero.embeddings.actor_mcts import ActorMCTS
from datetime import timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
# tf.compat.v1.logging.set_verbosity("CRITICAL")

moves_per_complexity = 4


def get_blocker(num_blockers=1, exclude_vars=[]):
    """Get a string of terms to place between target simplification terms
    in order to challenge the agent's ability to use commutative/associative
    rules to move terms around."""
    vars = get_rand_vars(num_blockers, exclude_vars)
    out_terms = []
    for i in range(num_blockers):
        out_terms.append("{}{}".format(maybe_int(), vars[i]))
    return " + ".join(out_terms)


def move_around_blockers_one(number_blockers):
    # two like terms separated by (n) blocker terms, e.g. 2 ~ "4x + (y + f) + x"
    var = rand_var()
    complexity = 2 + number_blockers
    blockers = get_blocker(number_blockers, [var])
    problem = "{}{} + {} + {}{}".format(maybe_int(), var, blockers, maybe_int(), var)
    return problem, complexity


def move_around_blockers_two(number_blockers):
    # two like terms with three blockers: "7a + 4x + (2f + j) + x + 3d"
    rand_vars = get_rand_vars(3)
    [one_var, two_var, three_var] = rand_vars
    complexity = 4 + number_blockers
    problem = "{}{} + {}{} + {} + {}{} + {}{}".format(
        maybe_int(),
        one_var,
        maybe_int(),
        two_var,
        get_blocker(number_blockers, rand_vars),
        maybe_int(),
        two_var,
        maybe_int(),
        three_var,
    )
    return problem, complexity


def move_around_interleaved_like_terms(number_terms, number_pairs):
    # interleaved multiple like variables: "4x + 2y + 6x + 3y"
    complexity = number_terms * number_pairs
    terms = []
    rand_vars = get_rand_vars(number_terms)
    for i in range(number_pairs):
        for j in range(number_terms):
            terms.append("{}{}".format(maybe_int(), rand_vars[j]))
    return " + ".join(terms), complexity


commutative_lessons = build_lesson_plan(
    "Mathy training (combine after reordering)",
    [
        LessonExercise(
            lesson_name="inner blockers",
            problem_count=2,
            problem_fn=lambda: move_around_blockers_one(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="inner blockers (difficult)",
            problem_count=2,
            problem_fn=lambda: move_around_blockers_one(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="outer blockers with innner blockers",
            problem_count=2,
            problem_fn=lambda: move_around_blockers_two(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="outer blockers with innner blockers (difficult)",
            problem_count=2,
            problem_fn=lambda: move_around_blockers_two(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="sets of 2 like terms interleaved",
            problem_count=2,
            problem_fn=lambda: move_around_interleaved_like_terms(2, 2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
    ],
)


commutative_lessons_two = build_lesson_plan(
    "Mathy training (combine after reordering)",
    [
        LessonExercise(
            lesson_name="inner blockers",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_one(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="inner blockers (difficult)",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_one(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="outer blockers with innner blockers",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_two(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="outer blockers with innner blockers (difficult)",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_two(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="3 sets of 2 like terms interleaved",
            problem_count=1,
            problem_fn=lambda: move_around_interleaved_like_terms(3, 2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
    ],
)

lesson_plan = build_lesson_plan(
    "Mathy training (combine like terms)",
    [
        LessonExercise(
            lesson_name="two terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=8,
            num_exploration_moves=8,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="three terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=12,
            num_exploration_moves=12,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="four terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=16,
            num_exploration_moves=16,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="five terms",
            problem_count=2,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=20,
            num_exploration_moves=20,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="six terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(6),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=24,
            num_exploration_moves=24,
            mcts_sims=500,
        ),
    ],
)
lesson_two = build_lesson_plan(
    "Mathy training (combine like terms)",
    [
        LessonExercise(
            lesson_name="five terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="six terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(6),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="eight terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(8),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="fifteen terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(15),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
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
    transfer_from=(
        "The name of another model to warm start this one from. Think Transfer Learning",
        "positional",
        None,
        str,
    ),
)
def main(model_dir, transfer_from=None):
    import tensorflow as tf

    eval_interval = 3
    eval_ltm_sample_size = 2048
    episode_counter = 0
    counter = 0
    controller = MathGame(verbose=True, focus_buckets=3)
    mathy = MathModel(controller.action_size, model_dir, init_model_dir=transfer_from)
    experience = MathExperience(mathy.model_dir)
    mathy.start()
    while True:
        print("[Lesson:{}]".format(counter))
        counter = counter + 1
        eval_run = bool(counter % eval_interval == 0)
        num_solved = 0
        num_failed = 0

        # plan = lesson_plan if counter % 2 == 0 else commutative_lessons
        plan = lesson_plan if counter % 2 == 0 else commutative_lessons
        if eval_run:
            print("\n\n=== Evaluating model with exploitation strategy ===")
            mathy.stop()
            mathy_eval = MathModel(
                controller.action_size,
                model_dir,
                init_model_dir=os.path.abspath(mathy.model_dir),
                # We want to initialize from the training model for each evaluation. (?)
                init_model_overwrite=True,
                is_eval_model=True,
            )
            eval_experience = MathExperience(mathy_eval.model_dir)
            mathy_eval.start()

        else:
            eval_experience = None
            # plan = lesson_plan if counter % 2 == 0 else commutative_lessons
            plan = lesson_plan if counter % 2 != 0 else commutative_lessons

        lessons = plan.lessons[:]
        # we fill this with episode rewards and when it's a fixed size we
        # dump the average value to tensorboard
        reward_sample_buffer = []
        while len(lessons) > 0:
            lesson = lessons.pop(0)
            controller.lesson = lesson
            print("\n{} - {}...".format(plan.name.upper(), lesson.name.upper()))
            for i in range(lesson.problem_count):
                env_state, complexity = controller.get_initial_state()
                complexity_value = complexity * moves_per_complexity
                controller.verbose = not eval_run
                if eval_run:
                    num_rollouts = 50
                    num_exploration_moves = 0
                    epsilon = 0
                else:
                    num_rollouts = lesson.mcts_sims
                    num_exploration_moves = (
                        lesson.num_exploration_moves
                        if lesson.num_exploration_moves is not None
                        else complexity_value
                    )
                    epsilon = 1.0
                controller.max_moves = (
                    lesson.max_turns
                    if lesson.max_turns is not None
                    else complexity_value
                )
                model = mathy_eval if eval_run else mathy
                mcts = MCTS(controller, model, epsilon, num_rollouts)
                actor = ActorMCTS(mcts, num_exploration_moves)
                final_result = None
                time_steps = []
                start = time.time()
                while final_result is None:
                    env_state, train_example, final_result = actor.step(
                        controller, env_state, model, time_steps
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
                reward_sample_buffer.append(episode_reward)
                if eval_experience is not None:
                    eval_experience.add_batch(episode_examples)
            if not eval_run:
                mathy.train(experience.short_term, experience.long_term)
            else:
                mathy_eval.train(eval_experience.short_term, eval_experience.long_term)

            print("writing avg reward")
            train_summary_writer = tf.summary.create_file_writer(model.model_dir)
            with train_summary_writer.as_default():
                episode_counter = episode_counter + 1
                tf.summary.experimental.set_step(episode_counter)
                tf.compat.v2.summary.scalar(
                    name="rewards/average_lesson_{}".format(
                        lesson.name.replace(" ", "_").lower()
                    ),
                    data=numpy.mean(reward_sample_buffer),
                    step=tf.summary.experimental.get_step(),
                )
            reward_sample_buffer = []

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
            print("training on evaluation data...")
            mathy_eval.stop()
            mathy_eval = None
            mathy.start()

    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
