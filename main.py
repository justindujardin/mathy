# coding: utf8
import json
import os
import random
import time
from datetime import timedelta

import numpy
import plac
import tensorflow as tf
from colr import color

from mathy.agent.controller import MathModel
from mathy.agent.curriculum.level1 import lessons
from mathy.agent.training.actor_mcts import ActorMCTS
from mathy.agent.training.math_experience import (
    MathExperience,
    balanced_reward_experience_samples,
)
from mathy.agent.training.mcts import MCTS
from mathy.math_game import MathGame

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "5"
tf.compat.v1.logging.set_verbosity("CRITICAL")


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
    lesson_id=("The lesson plan to execute by ID", "option", "l", str),
    learning_rate=("The learning rate to use when training", "option", "lr", float),
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
def main(
    model_dir,
    transfer_from=None,
    lesson_id=None,
    initial_train=False,
    verbose=False,
    learning_rate=3e-4,
):
    global lessons
    shuffle_lessons = False
    min_train_experience = 256
    eval_interval = 2
    short_term_size = 768
    long_term_size = 8192
    initial_train_iterations = 10
    episode_counter = 0
    counter = 0
    training_epochs = 8
    controller = MathGame(verbose=True)
    mathy = MathModel(
        controller.action_size,
        model_dir,
        init_model_dir=transfer_from,
        learning_rate=learning_rate,
        long_term_size=long_term_size,
        epochs=training_epochs,
    )
    experience = MathExperience(mathy.model_dir, short_term_size)
    mathy.start()
    if lesson_id is None:
        plan = lessons[list(lessons)[0]]
    elif lesson_id not in lessons:
        raise ValueError(
            f"[lesson] ERROR: '{lesson_id}' not found in ids. Valid lessons are: {', '.join(lessons)} "
        )
    else:
        plan = lessons[lesson_id]

    if initial_train is True:
        print(
            color(
                "[training] {} iterations before beginning self-practice".format(
                    initial_train_iterations
                ),
                fore="blue",
            )
        )
        old = mathy.epochs
        mathy.epochs = initial_train_iterations
        mathy.train(experience.short_term, experience.long_term, train_all=True)
        mathy.epochs = old
        print(color("Okay, let's do this!", fore="green"))

    while True:
        print(f"[lesson] session {counter}")
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
                controller.action_size,
                model_dir,
                init_model_dir=os.path.abspath(mathy.model_dir),
                # We want to initialize from the training model for each evaluation. (?)
                init_model_overwrite=True,
                is_eval_model=True,
                learning_rate=learning_rate,
                epochs=training_epochs,
                long_term_size=long_term_size,
            )
            eval_experience = MathExperience(mathy_eval.model_dir, short_term_size=256)
            mathy_eval.start()

        else:
            eval_experience = None
        model = mathy_eval if eval_run else mathy

        lessons = plan.lessons[:]

        if shuffle_lessons:
            random.shuffle(lessons)
        print("lesson order: {}".format([l.name for l in lessons]))
        # we fill this with episode rewards and when it's a fixed size we
        # dump the average value to tensorboard
        ep_reward_buffer = []
        while len(lessons) > 0:
            lesson = lessons.pop(0)
            controller.lesson = lesson
            print("\n{} - {}...".format(plan.name.upper(), lesson.name.upper()))
            # Fill up a certain amount of experience per problem type
            lesson_experience_count = 0
            if lesson.num_observations is not None:
                iter_experience = lesson.num_observations
            else:
                iter_experience = short_term_size
            while lesson_experience_count < iter_experience:
                env_state, complexity = controller.get_initial_state(
                    print_problem=False
                )
                complexity_value = complexity * 4
                controller.verbose = eval_run or verbose
                if eval_run:
                    num_rollouts = 500
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
                if eval_experience is not None:
                    eval_experience.add_batch(episode_examples)

            # Train if we have enough data
            if experience.count > min_train_experience:
                if not eval_run:
                    model.train(
                        experience.short_term,
                        experience.long_term,
                        sampling_fn=balanced_reward_experience_samples,
                    )
                else:
                    model.train(
                        eval_experience.short_term,
                        experience.long_term,
                        sampling_fn=balanced_reward_experience_samples,
                    )
            else:
                print(
                    color(
                        f"[skip training] only have {experience.count} observations, but need at least {min_train_experience} before training",
                        fore="yellow",
                        style="bright",
                    )
                )
                continue

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
