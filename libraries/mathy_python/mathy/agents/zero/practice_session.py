from json import JSONEncoder
from typing import Any, List
import json
import os
import sys
import tempfile
import time
import types
from collections import deque
from multiprocessing import Array, Pool, Process, Queue, Value, cpu_count
from pathlib import Path
from pickle import Pickler, Unpickler
from random import shuffle
from shutil import copyfile
from .config import SelfPlayConfig
import numpy as np

import srsly
from .types import EpisodeHistory, EpisodeSummary
from ...agents.mcts import MCTS
from .lib.progress.bar import Bar
from .lib.average_meter import AverageMeter

INPUT_EXAMPLES_FILE_NAME = "examples.jsonl"


class PracticeSession:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    all_examples: List[EpisodeHistory] = []

    def has_examples(self) -> bool:
        return bool(len(self.all_examples) > 0)

    def __init__(self, runner, config: SelfPlayConfig, env_name: str):
        self.runner = runner
        self.config = config
        self.env = env_name
        self.problem_count = self.runner.config.self_play_problems
        self.all_examples = []
        self.skip_first_self_play = False
        loaded = self.load_training_examples()
        if loaded is not False:
            print("Loaded examples from: {}".format(loaded))

    def learn(self):
        iterations = self.problem_count
        for i in range(1, self.config.training_iterations + 1):
            solve, fail, summary, new_examples = self.run_self_play(i, iterations)
            self.run_network_training(i)

    def run_self_play(self, iteration, num_episodes):
        bar = Bar(self.env, max=num_episodes)
        bar.suffix = "working on first problem..."
        bar.next()
        current_episode = 0

        solved = 0
        failed = 0

        def episode_complete(self, episode, summary):
            nonlocal current_episode, bar, num_episodes, solved, failed
            current_episode += 1
            if summary.solved is True:
                solved = solved + 1
            else:
                failed = failed + 1

            bar.message = "{} [{},{}]".format(self.env.upper(), solved, failed)
            bar.suffix = "({eps}/{maxeps}) | total: {total:} | remaining: {eta:}".format(
                eps=current_episode,
                maxeps=num_episodes,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()

        episodes_with_args = []
        for i in range(1, num_episodes + 1):
            episodes_with_args.append(dict(model_dir=self.runner.config.model_dir))

        old_update = self.runner.episode_complete
        self.runner.episode_complete = types.MethodType(episode_complete, self)
        new_examples, episodes_rewards = self.runner.execute_episodes(
            episodes_with_args
        )
        # Output a few solve/fail stats
        complexity_stats = dict()
        solve = 0
        fail = 0

        def add_stat(complexity, value):
            nonlocal complexity_stats, solve, fail
            if complexity not in complexity_stats:
                complexity_stats[complexity] = dict(solve=0, fail=0)
            if value >= 0:
                solve += 1
                complexity_stats[complexity]["solve"] += 1
            elif value < 0:
                fail += 1
                complexity_stats[complexity]["fail"] += 1

        summary = None
        for summary in episodes_rewards:
            value = summary.reward
            complexity = str(summary.complexity)
            add_stat(complexity, value)
        print(
            "\n\nPractice results:\n --- Solved ({}) --- Failed ({})".format(
                solve, fail
            )
        )
        print("By complexity:\n{}\n\n".format(json.dumps(complexity_stats, indent=2)))
        self.runner.episode_complete = old_update
        bar.finish()
        self.all_examples.extend(new_examples)
        self.save_training_examples()
        return solve, fail, summary, new_examples

    def run_network_training(self, iteration):
        """
        Train the current best model with the gathered training examples and return
        a tuple of (best, new) where best is the existing best trained model (or a blank
        one) and new is the model that was just trained.
        """
        train_examples = self.all_examples
        # print(train_examples)
        print("Training with {} examples".format(len(train_examples)))
        return self.runner.train(
            iteration, train_examples, self.runner.config.model_dir
        )

    def load_training_examples(self):
        file_path = Path(self.runner.config.model_dir or "") / INPUT_EXAMPLES_FILE_NAME
        if not file_path.is_file():
            return False
        examples = list(srsly.read_jsonl(str(file_path)))
        # with file_path.open("r", encoding="utf8") as f:
        #     for line in f:
        #         ex = ujson.loads(line)
        #         examples.append(ex)
        self.all_examples = examples
        self.skip_first_self_play = True
        return str(file_path)

    def save_training_examples(self):
        model_dir = Path(self.runner.config.model_dir)
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)

        # Write to local then copy (don't thrash virtual file systems like GCS)
        _, tmp_file = tempfile.mkstemp()
        srsly.write_jsonl(tmp_file, self.all_examples)
        out_file = model_dir / INPUT_EXAMPLES_FILE_NAME
        copyfile(tmp_file, str(out_file))
        os.remove(tmp_file)
        return str(out_file)
