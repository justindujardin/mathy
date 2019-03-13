from mathzero.training.problems import MODE_SIMPLIFY_POLYNOMIAL, simplify_multiple_terms
from mathzero.training.lessons import build_lesson_plan, LessonExercise

lessons = build_lesson_plan(
    "Polynomials (add/mult)",
    [
        LessonExercise(
            lesson_name="Two terms",
            problem_count=200,
            problem_fn=lambda: simplify_multiple_terms(2),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=10,
            mcts_sims=250,
        ),
        LessonExercise(
            lesson_name="Three variable terms",
            problem_count=100,
            problem_fn=lambda: simplify_multiple_terms(3),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=15,
            mcts_sims=300,
        ),
        LessonExercise(
            lesson_name="Four variable terms",
            problem_count=24,
            problem_fn=lambda: simplify_multiple_terms(4),
            problem_type=MODE_SIMPLIFY_POLYNOMIAL,
            max_turns=25,
            mcts_sims=750,
        ),
    ],
)
