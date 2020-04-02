from mathy.teacher import Student, Teacher

topic_names = ["poly", "binomial", "poly-blockers", "complex"]
me = 0


def test_teacher_env_rotation_by_iteration_modulus():
    teacher = Teacher(topic_names, num_students=2)

    # Test env rotation
    for i, e in enumerate(topic_names):
        # rotation is modulus by iteration
        result = teacher.get_env(1, i)
        assert result == f"mathy-{e}-easy-v0"


def test_teacher_evaluation_window_win_loss_record():
    teacher = Teacher(topic_names, eval_window=10)
    # Report win/loss
    teacher.report_result(me, 1.0)
    teacher.report_result(me, -0.1)
    teacher.report_result(me, 1.0)

    student: Student = teacher.students[me]
    assert student.topics[student.topic].positives == 2
    assert student.topics[student.topic].negatives == 1


def test_teacher_evaluation_objects():
    teacher = Teacher(topic_names, eval_window=10)
    # Report win/loss
    teacher.report_result(me, 1.0)
    teacher.report_result(me, -0.1)
    teacher.report_result(me, 1.0)

    student: Student = teacher.students[me]
    assert student.topics[student.topic].positives == 2
    assert student.topics[student.topic].negatives == 1


def test_teacher_evaluation_window_change_difficulty():
    teacher = Teacher(
        topic_names, eval_window=10, win_threshold=0.7, lose_threshold=0.5
    )
    # Report win/loss
    student = teacher.students[me]

    assert student.topics[student.topic].difficulty == "easy"

    # 60% wins remains at the same difficulty
    for i in range(4):
        teacher.report_result(me, -1.0)
    for i in range(6):
        teacher.report_result(me, 1.0)

    assert student.topics[student.topic].difficulty == "easy"

    # >= 70% wins goes to higher difficulty
    for i in range(7):
        teacher.report_result(me, 1.0)
    for i in range(3):
        teacher.report_result(me, -1.0)

    assert student.topics[student.topic].difficulty == "normal"

    # <= 50% wins goes to easier difficulty
    for i in range(6):
        teacher.report_result(me, -1.0)
    for i in range(4):
        teacher.report_result(me, 1.0)

    assert student.topics[student.topic].difficulty == "easy"
