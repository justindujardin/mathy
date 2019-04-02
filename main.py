# coding: utf8
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


quick_test_plan = build_lesson_plan(
    "dev_test",
    [
        LessonExercise(
            lesson_name="two_terms",
            problem_count=2,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=50,
        ),
    ],
)

lesson_plan = build_lesson_plan(
    "combine_like_terms_1",
    [
        LessonExercise(
            lesson_name="two_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="four_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="five_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="six_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(6),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="eight_terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(8),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="inner_blockers",
            problem_count=6,
            problem_fn=lambda: move_around_blockers_one(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="inner_blockers_difficult",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_one(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="outer_inner_blockers",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_two(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="outer_inner_blockers_difficult",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_two(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="interleaved_like_terms",
            problem_count=1,
            problem_fn=lambda: move_around_interleaved_like_terms(2, 2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
    ],
)

lesson_quick = build_lesson_plan(
    "combine_like_terms_1",
    [
        LessonExercise(
            lesson_name="two_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="three_terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="inner_blockers",
            problem_count=6,
            problem_fn=lambda: move_around_blockers_one(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="outer_inner_blockers",
            problem_count=1,
            problem_fn=lambda: move_around_blockers_two(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="interleaved_like_terms",
            problem_count=1,
            problem_fn=lambda: move_around_interleaved_like_terms(2, 2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            mcts_sims=500,
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
    initial_train=(
        "When true, train the network on everything in `examples.json` in the checkpoint directory",
        "flag",
        "t",
    ),
    verbose=(
        "When true, print all problem moves rather than just during evaluation",
        "flag",
        "v",
    ),
)
def main(model_dir, transfer_from=None, initial_train=False, verbose=False):
    import tensorflow as tf

    eval_interval = 2
    initial_train_iterations = 10
    eval_ltm_sample_size = 2048
    episode_counter = 0
    counter = 0
    controller = MathGame(verbose=True)
    mathy = MathModel(controller.action_size, model_dir, init_model_dir=transfer_from)
    experience = MathExperience(mathy.model_dir, 256)
    mathy.start()

    if initial_train is True:
        print(
            color(
                "Training for {} iterations on existing knowledge before beginning class".format(
                    initial_train_iterations
                ),
                fore="blue",
            )
        )
        old = mathy.args.epochs
        mathy.args.epochs = 10
        mathy.train(experience.short_term, experience.long_term, train_all=True)
        mathy.args.epochs = old
        print(color("Okay, let's do this!", fore="green"))

    while True:
        print("[Lesson:{}]".format(counter))
        counter = counter + 1
        eval_run = bool(counter % eval_interval == 0)
        num_solved = 0
        num_failed = 0

        plan = lesson_quick
        # plan = lesson_plan if counter % 5 == 0 else commutative_lessons
        # plan = quick_test_plan
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
        model = mathy_eval if eval_run else mathy

        lessons = plan.lessons[:]
        # we fill this with episode rewards and when it's a fixed size we
        # dump the average value to tensorboard
        ep_reward_buffer = []
        while len(lessons) > 0:
            lesson = lessons.pop(0)
            controller.lesson = lesson
            print("\n{} - {}...".format(plan.name.upper(), lesson.name.upper()))
            for i in range(lesson.problem_count):
                env_state, complexity = controller.get_initial_state(
                    print_problem=False
                )
                complexity_value = complexity * moves_per_complexity
                controller.verbose = eval_run or verbose
                if eval_run:
                    num_rollouts = 150
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
                # generate a new problem now that we've set the max_turns
                env_state, complexity = controller.get_initial_state()
                model = mathy_eval if eval_run else mathy
                mcts = MCTS(controller, model, epsilon, num_rollouts)
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
                if eval_experience is not None:
                    eval_experience.add_batch(episode_examples)
            if not eval_run:
                model.train(experience.short_term, experience.long_term)
            else:
                model.train(eval_experience.short_term, experience.long_term)

            summary_writer = tf.summary.create_file_writer(model.model_dir)
            with summary_writer.as_default():
                global_step = model.network.get_variable_value("global_step")
                var_name = "{}/step_avg_reward_{}".format(
                    plan.name.replace(" ", "_").lower(),
                    lesson.name.replace(" ", "_").lower(),
                )
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
            print("training on evaluation data...")
            mathy_eval.stop()
            mathy_eval = None
            mathy.start()

    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
