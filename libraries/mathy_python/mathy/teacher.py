import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StudentEvaluation(BaseModel):
    env: str
    total: int
    solved: int
    failed: int


class Topic(BaseModel):
    # The topic name used to interpolate the env name
    name: str
    # The current topic difficulty
    difficulty: str = "easy"
    # The number of observations for the current eval window
    count: int = 0
    # The current positives count
    positives: int = 0
    # The current negatives count
    negatives: int = 0

    def reset_counts(self):
        self.positives = 0
        self.negatives = 0
        self.count = 0


class Student(BaseModel):
    # The index of the student in the students list
    id: int
    # Current topic of study
    topic: str
    # The current difficulty settings for each env
    topics: Dict[str, Topic]


class Teacher:
    
    students: List[Student]

    def __init__(
        self,
        topic_names: List[str],
        num_students: int = 1,
        difficulty: Optional[str] = None,
        eval_window: int = 50,
        win_threshold: float = 0.95,
        lose_threshold: float = 0.34,
    ):
        self.topic_names = topic_names
        self.eval_window = eval_window
        self.win_threshold = win_threshold
        self.lose_threshold = lose_threshold
        self.difficulty = difficulty
        if self.difficulty is not None:
            print(f"difficulty will not adjust and is fixed to: {self.difficulty}")
        self.initialize_students(num_students)

    def initialize_students(self, num_students: int):
        self.num_students = num_students
        self.students = []
        for i in range(self.num_students):
            student_topics = {}
            start_topic = random.choice(self.topic_names)
            for topic in self.topic_names:
                difficulty = self.difficulty if self.difficulty is not None else "easy"
                student_topics[topic] = Topic(name=topic, difficulty=difficulty)
            self.students.append(
                Student(id=i, topic=start_topic, topics=student_topics)
            )

    def previous_difficulty(self, difficulty: str) -> str:
        """Return the previous difficulty level given an input difficulty.
        
        # Arguments
        difficulty (str): The difficulty to decrease

        # Returns
        (str): The difficulty level before the input, if any.
        """
        if difficulty == "hard":
            return "normal"
        elif difficulty == "normal":
            return "easy"
        return "easy"

    def next_difficulty(self, difficulty: str) -> str:
        """Return the previous difficulty level given an input difficulty.
        
        # Arguments
        difficulty (str): The difficulty to increase

        # Returns
        (str): The difficulty level after the input, if any.
        """
        if difficulty == "easy":
            return "normal"
        elif difficulty == "normal":
            return "hard"
        return "hard"

    def report_result(
        self, student_index: int, reward: float, data: Any = None
    ) -> Optional[float]:
        student = self.students[student_index]
        topic: Topic = student.topics[student.topic]
        if reward > 0.0:
            topic.positives += 1
        else:
            topic.negatives += 1
        topic.count += 1

        if topic.count >= self.eval_window:
            win_ratio = topic.positives / self.eval_window
            action = "kept at the same difficulty, to gather more experience"
            # If the difficulty is locked, don't adjust it.
            if self.difficulty is not None:
                pass
            elif win_ratio >= self.win_threshold:
                topic.difficulty = self.next_difficulty(topic.difficulty)
                action = "promoted"
            elif win_ratio <= self.lose_threshold:
                topic.difficulty = self.previous_difficulty(topic.difficulty)
                action = "demoted"
            if student_index == 0:
                pct = int(win_ratio * 100)
                type = topic.name
                diff = topic.difficulty
                print(
                    f"Solved {pct}% of {type} problems and was {action}. "
                    f"Next round will use {diff} difficulty problems."
                )
            topic.reset_counts()
            return win_ratio
        return None

    def get_env(self, student_index: int, iteration: int) -> str:
        """Get the current environment a student should be using.

        # Arguments
        student_index (int): The index of the student in `self.students` array.
        iteration (int): The current iteration (usually an episode).

        # Returns
        (str): The name of a mathy environment to use.
        """
        student = self.students[student_index]
        # The console printing student is special, it trains in everything
        len_topics = len(self.topic_names)
        student.topic = self.topic_names[iteration % len_topics]
        topic = student.topics[student.topic]
        return f"mathy-{topic.name}-{topic.difficulty}-v0"
