import json
from pathlib import Path
from mathy.agent.training.practice_session import PracticeSession
from mathy.mathy_env import MathyEnv
from mathy.agent.controller import MathModel
from mathy.agent.training.practice_runner import (
    PracticeRunner,
    ParallelPracticeRunner,
    RunnerConfig,
)
from mathy.core.parser import ExpressionParser, ParserException
from .lessons import LessonPlan, LessonExercise


def lesson_runner(
    model_dir, lesson_plan, parallel=True, dev_mode=False, skip_completed=True
):
    """Practice a concept for up to (n) lessons or until the concept is learned as defined
    by the lesson plan. """
    lessons = lesson_plan.lessons[:]
    lesson_checkpoint = Path("{}lesson_state.json".format(model_dir))
    BaseEpisodeRunner = PracticeRunner if not parallel else ParallelPracticeRunner

    # If we're resuming a training session, start at the lesson we left off with last time
    if lesson_checkpoint.is_file() and skip_completed is True:
        with lesson_checkpoint.open("r", encoding="utf8") as f:
            lesson_state = json.loads(f.read())
    else:
        lesson_state = {"completed": {}}
    completed_lessons = lesson_state["completed"]
    out_lessons = []
    for lesson in lessons:
        if lesson.name in completed_lessons:
            print(
                "-- skipping '{}' because it was completed earlier".format(lesson.name)
            )
        else:
            out_lessons.append(lesson)
    lessons = out_lessons

    # Copy the lessons array so we can pop from it as the agent progresses in competency
    if not isinstance(lesson_plan, LessonPlan):
        raise ValueError(
            "Lesson plan must be a LessonPlan instance. Try calling 'build_lesson_plan'"
        )
    while len(lessons) > 0:
        lesson = lessons.pop(0)

        class LessonRunner(BaseEpisodeRunner):
            def get_game(self):
                return MathyEnv(
                    verbose=dev_mode, lesson=lesson, max_moves=lesson.max_turns
                )

            def get_predictor(self, game, all_memory=False):
                return MathModel(game, model_dir, all_memory)

        num_to_advance = 1
        progress_counter = 0

        def session_done(lesson_exercise, num_solved, num_failed):
            # NOTE: Trying rotating through all the challenges rather than overfitting on one
            return False
            # nonlocal num_to_advance, progress_counter
            # if num_failed == 0:
            #     progress_counter = progress_counter + 1
            # else:
            #     progress_counter = 0
            # if progress_counter == num_to_advance:
            #     print(
            #         "\n\n{} COMPLETED! [get all the problems right {} times in a row]\n\n".format(
            #             lesson_exercise.name, num_to_advance
            #         )
            #     )
            #     progress_counter = 0
            #     return False

        print("Practicing {} - {}...".format(lesson_plan.name, lesson.name))
        args = {"self_play_iterations": lesson.problem_count}
        config = RunnerConfig(
            model_dir=model_dir,
            num_mcts_sims=lesson.mcts_sims,
            num_exploration_moves=lesson.num_exploration_moves,
            cpuct=1.0,
        )
        runner = LessonRunner(config)
        c = PracticeSession(runner, lesson)
        c.learn(session_done)
        lesson_state["completed"][lesson.name] = True
        write_lesson_state(lesson_checkpoint, lesson_state)


def write_lesson_state(file_path, file_data):
    with Path(file_path).open("w", encoding="utf-8") as f:
        f.write(json.dumps(file_data))
