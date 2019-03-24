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
)
from mathzero.embeddings.math_experience import MathExperience
from mathzero.training.mcts import MCTS
from mathzero.embeddings.actor_mcts import ActorMCTS
from datetime import timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
# tf.compat.v1.logging.set_verbosity("CRITICAL")

moves_per_complexity = 3

eval_plan = build_lesson_plan(
    "Mathy (exploitation)",
    [
        LessonExercise(
            lesson_name="two terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=8,
            num_exploration_moves=0,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="three terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=12,
            num_exploration_moves=0,
            mcts_sims=50,
        ),
        LessonExercise(
            lesson_name="four terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=16,
            num_exploration_moves=0,
            mcts_sims=50,
        ),
        LessonExercise(
            lesson_name="five terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=20,
            num_exploration_moves=50,
            mcts_sims=50,
        ),
        LessonExercise(
            lesson_name="six terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(6),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=24,
            num_exploration_moves=0,
            mcts_sims=150,
        ),
        LessonExercise(
            lesson_name="seven terms",
            problem_count=3,
            problem_fn=lambda: simplify_multiple_terms(7),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=35,
            num_exploration_moves=0,
            mcts_sims=150,
        ),
        LessonExercise(
            lesson_name="eight terms",
            problem_count=3,
            problem_fn=lambda: simplify_multiple_terms(8),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=40,
            num_exploration_moves=0,
            mcts_sims=150,
        ),
    ],
)


def get_rand_vars(num_vars, exclude_vars=[]):
    """Get a list of random variables, excluding the given list of hold-out variables"""
    var = rand_var()
    if num_vars > 25:
        raise ValueError("out of range: there are only twenty-six variables")
    rand_vars = set(rand_var())
    while len(rand_vars) < num_vars:
        _rand = rand_var()
        if _rand not in exclude_vars:
            rand_vars.add(rand_var())
    return list(rand_vars)


def build_commutative_lessons():
    t_one_blocker = "one_blocker_three_terms"
    t_two_like_terms_blocking = "two_like_terms_blocking"
    t_two_blockers = "two_blockers_four_terms"
    t_three_blockers = "three_blockers_five_terms"
    t_four_blockers = "four_blockers_six_terms"
    t_four_with_n_blockers = "n_blockers"

    def get_blocker(num_blockers=1, exclude_vars=[]):
        vars = get_rand_vars(num_blockers, exclude_vars)
        out_terms = []
        for i in range(num_blockers):
            out_terms.append("{}{}".format(maybe_int(), vars[i]))
        return " + ".join(out_terms)

    def commute_problems(problem_type, extra_n=None):
        """Commuting lesson plan"""
        rand_vars = set(rand_var())
        while len(rand_vars) < 5:
            rand_vars.add(rand_var())
        rand_vars = list(rand_vars)
        random.shuffle(rand_vars)
        one_var = rand_vars[0]
        two_var = rand_vars[1]
        three_var = rand_vars[2]
        four_var = rand_vars[3]
        five_var = rand_vars[4]
        if problem_type == t_one_blocker:
            # move around one term: "4x + 2y + 6x"
            complexity = 3
            problem = "{}{} + {}{} + {}{}".format(
                maybe_int(), one_var, maybe_int(), two_var, maybe_int(), one_var
            )
        elif problem_type == t_two_like_terms_blocking:
            # interleaved multiple like variables: "4x + 2y + 6x + 3y"
            complexity = 4
            problem = "{}{} + {}{} + {}{} + {}{}".format(
                maybe_int(),
                one_var,
                maybe_int(),
                two_var,
                maybe_int(),
                one_var,
                maybe_int(),
                two_var,
            )
        elif problem_type == t_two_blockers:
            # two like terms with two blockers: "4x + y + 6x + 3z"
            complexity = 4
            problem = "{}{} + {}{} + {}{} + {}{}".format(
                maybe_int(),
                one_var,
                maybe_int(),
                two_var,
                maybe_int(),
                one_var,
                maybe_int(),
                three_var,
            )
        elif problem_type == t_three_blockers:
            # two like terms with three blockers: "7w + 4x + 2y + x + 3z"
            complexity = 5
            problem = "{}{} + {}{} + {}{} + {}{} + {}{}".format(
                maybe_int(),
                four_var,
                maybe_int(),
                one_var,
                maybe_int(),
                two_var,
                maybe_int(),
                one_var,
                maybe_int(),
                three_var,
            )
        elif problem_type == t_four_blockers:
            # two like terms with three blockers: "7w + 4x + 2y + y + x + 3z"
            complexity = 5
            problem = "{}{} + {}{} + {}{} + {}{} + {}{} + {}{}".format(
                maybe_int(),
                four_var,
                maybe_int(),
                one_var,
                maybe_int(),
                two_var,
                maybe_int(),
                five_var,
                maybe_int(),
                one_var,
                maybe_int(),
                three_var,
            )
        elif problem_type == t_four_with_n_blockers:
            if extra_n is None:
                raise ValueError("must pass extra_n as the number of blockers to add")
            # two like terms with three blockers: "7w + 4x + 2y + y + x + 3z"
            complexity = 4 + extra_n
            problem = "{}{} + {}{} + {} + {}{} + {}{}".format(
                maybe_int(),
                four_var,
                maybe_int(),
                one_var,
                get_blocker(extra_n),
                maybe_int(),
                one_var,
                maybe_int(),
                three_var,
            )
        else:
            raise ValueError("unknown complexity for commute tests")
        return problem, complexity

    return [
        LessonExercise(
            lesson_name=t_one_blocker,
            problem_count=2,
            problem_fn=lambda: commute_problems(t_one_blocker),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=3 * moves_per_complexity,
            num_exploration_moves=3 * moves_per_complexity,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name=t_two_blockers,
            problem_count=2,
            problem_fn=lambda: commute_problems(t_two_blockers),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=4 * moves_per_complexity,
            num_exploration_moves=4 * moves_per_complexity,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name=t_two_like_terms_blocking,
            problem_count=2,
            problem_fn=lambda: commute_problems(t_two_like_terms_blocking),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=4 * moves_per_complexity,
            num_exploration_moves=4 * moves_per_complexity,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name=t_three_blockers,
            problem_count=2,
            problem_fn=lambda: commute_problems(t_three_blockers),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=5 * moves_per_complexity,
            num_exploration_moves=5 * moves_per_complexity,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name=t_four_blockers,
            problem_count=2,
            problem_fn=lambda: commute_problems(t_four_blockers),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=6 * moves_per_complexity,
            num_exploration_moves=6 * moves_per_complexity,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name=t_four_with_n_blockers,
            problem_count=2,
            problem_fn=lambda: commute_problems(t_four_with_n_blockers, 3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=7 * moves_per_complexity,
            num_exploration_moves=7 * moves_per_complexity,
            mcts_sims=500,
        ),
    ]


commutative_lessons = build_lesson_plan(
    "Mathy training (combine after reordering)", build_commutative_lessons()
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
            problem_count=1,
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
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(5),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=20,
            num_exploration_moves=20,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="six terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(6),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=24,
            num_exploration_moves=24,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="eight terms",
            problem_count=4,
            problem_fn=lambda: simplify_multiple_terms(8),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=moves_per_complexity * 8,
            num_exploration_moves=moves_per_complexity * 8,
            mcts_sims=500,
        ),
        LessonExercise(
            lesson_name="fifteen terms",
            problem_count=1,
            problem_fn=lambda: simplify_multiple_terms(15),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=moves_per_complexity * 12,
            num_exploration_moves=moves_per_complexity * 12,
            mcts_sims=150,
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
    eval_interval = 5
    counter = 0
    dev_mode = True
    controller = MathGame(verbose=dev_mode, focus_buckets=3)
    experience = MathExperience(model_dir)
    mathy = MathModel(controller.action_size, model_dir, init_model_dir=transfer_from)
    mathy.start()
    while True:
        print("[Lesson:{}]".format(counter))
        counter = counter + 1
        eval_run = bool(counter % eval_interval == 0)
        eval_solve = 0
        eval_fail = 0

        if eval_run:
            print("\n\n=== Evaluating model ===")
            plan = eval_plan
        else:
            # plan = lesson_plan if counter % 2 == 0 else commutative_lessons
            plan = lesson_two if counter % 2 != 0 else commutative_lessons
        lessons = plan.lessons[:]
        while len(lessons) > 0:
            lesson = lessons.pop(0)
            controller.lesson = lesson
            controller.max_moves = lesson.max_turns
            print("\n{} - {}...".format(plan.name.upper(), lesson.name.upper()))

            for i in range(lesson.problem_count):
                env_state, complexity = controller.get_initial_state()
                mcts = MCTS(controller, mathy, 1.0, lesson.mcts_sims)
                actor = ActorMCTS(mcts, lesson.num_exploration_moves)
                final_result = None
                time_steps = []
                start = time.time()
                while final_result is None:
                    env_state, train_example, final_result = actor.step(
                        controller, env_state, mathy, time_steps
                    )

                elapsed = time.time() - start
                episode_examples, episode_reward, is_win = final_result
                if is_win:
                    eval_solve = eval_solve + 1
                    outcome = "solved"
                    fore = "green"
                else:
                    eval_fail = eval_fail + 1
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
            # Eval runs only train after all problems are done
            if not eval_run:
                mathy.train(experience.short_term, experience.long_term)

        if eval_run:
            print(
                color(
                    "\n\n=== Evaluation complete solve({}) fail({}) ===\n\n".format(
                        eval_solve, eval_fail
                    ),
                    fore="blue",
                    style="bright",
                )
            )
            mathy.train(experience.short_term, experience.long_term)

    print("Complete. Bye!")
    mathy.stop()


if __name__ == "__main__":
    plac.call(main)
