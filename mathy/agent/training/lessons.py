from mathy.core.parser import ExpressionParser, ParserException

CONCEPT_ARITHMETIC = "arithmetic"
CONCEPT_COMBINE_TERMS = "combine_terms"


class LessonExercise:
    """A configuration for generating/practicing problems and evaluating competency at solving them"""

    def __init__(
        self,
        lesson_name,
        problem_fn,
        problem_type,
        problem_count,
        max_turns=None,
        mcts_sims=None,
        num_exploration_moves=None,
        training_wheels=False,
        num_observations=None,
    ):
        self.name = lesson_name
        self.problem_fn = problem_fn
        self.problem_type = problem_type
        self.problem_count = problem_count
        self.max_turns = max_turns
        self.mcts_sims = mcts_sims
        self.training_wheels = training_wheels
        self.num_exploration_moves = num_exploration_moves
        self.num_observations = num_observations


class LessonPlan:
    """A collection of lesson configurations grouped together under a concept name"""

    def __init__(self, plan_name, lessons):
        self.name = plan_name
        self.lessons = lessons


def build_lesson_plan(group_name, lessons):
    parser = ExpressionParser()
    for i, lesson in enumerate(lessons):
        if not isinstance(lesson, LessonExercise):
            raise ValueError("array should contain only LessonExercise class instances")
        for i in range(lesson.problem_count):
            try:
                problem = None
                problem, complexity = lesson.problem_fn()
                parser.parse(problem)
            except (ParserException, TypeError) as e:
                raise ValueError(
                    "The problem_fn for lesson ({} - {}) produced an invalid problem: {}".format(
                        i, group_name, problem
                    )
                )
    return LessonPlan(group_name, lessons)

