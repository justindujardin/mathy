import ujson
import sys
import types
from collections import deque
from .Arena import Arena
from .MCTS import MCTS
import numpy as np
import json
from .pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pathlib import Path
from pickle import Pickler, Unpickler
from random import shuffle
import os
from multiprocessing import Pool, Array, Process, Queue, cpu_count, Value
from .Game import Game

INPUT_EXAMPLES_FILE_NAME = "examples.jsonl"


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def has_examples(self) -> bool:
        return bool(len(self.all_examples) > 0)

    def __init__(self, runner, args=None):
        if args is None:
            args = dict()
        self.runner = runner
        self.training_iterations = args.get("training_iterations", 50)
        self.self_play_iterations = args.get("self_play_iterations", 100)
        self.model_win_loss_ratio = args.get("model_win_loss_ratio", 0.6)
        self.max_training_examples = args.get("max_training_examples", 200000)
        self.model_arena_iterations = args.get("model_arena_iterations", 30)
        self.all_examples = []
        self.skip_first_self_play = False
        loaded = self.load_training_examples()
        if loaded is not False:
            print("Loaded examples from: {}".format(loaded))

    def learn(self):
        """
        Performs training_iterations iterations with self_play_iterations episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= model_win_loss_ratio fraction of games.
        """
        iterations = self.self_play_iterations

        for i in range(1, self.training_iterations + 1):
            print("Session {}".format(i))
            self.run_self_play(i, iterations)
            self.save_training_examples()
            self.run_network_training(i)

    def run_self_play(self, iteration, num_episodes):
        if iteration == 1 and self.skip_first_self_play:
            return []
        bar = Bar("Practicing", max=num_episodes)
        bar.suffix = "working on first problem..."
        bar.next()
        current_episode = 0

        def episode_complete(self, episode, summary):
            nonlocal current_episode, bar, num_episodes
            current_episode += 1
            bar.suffix = "({eps}/{maxeps}) | Total: {total:} | ETA: {eta:}".format(
                eps=current_episode,
                maxeps=num_episodes,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()

        episodes_with_args = []
        for _ in range(1, num_episodes + 1):
            episodes_with_args.append(dict(model=self.runner.config.model_dir))

        old_update = self.runner.episode_complete
        self.runner.episode_complete = types.MethodType(episode_complete, self)
        new_examples, episode_rewards = self.runner.execute_episodes(episodes_with_args)
        # Output a few solve/fail stats

        complexity_stats = dict()
        solve = 0
        fail = 0

        def add_stat(complexity, value):
            nonlocal complexity_stats, solve, fail
            if complexity not in complexity_stats:
                complexity_stats[complexity] = dict(solve=0, fail=0)
            if value == 1:
                solve += 1
                complexity_stats[complexity]["solve"] += 1
            elif value == -1:
                fail += 1
                complexity_stats[complexity]["fail"] += 1
            pass

        for summary in episode_rewards:
            value = summary["reward"]
            complexity = str(summary["complexity"])
            add_stat(complexity, value)
        print(
            "\n\nPractice results:\n --- Solved ({}) --- Failed ({})".format(
                solve, fail
            )
        )
        print("By complexity:\n{}\n\n".format(json.dumps(complexity_stats, indent=2)))
        training_examples = deque(new_examples, maxlen=self.max_training_examples)
        self.runner.episode_complete = old_update
        bar.finish()
        self.all_examples.extend(training_examples)
        self.save_training_examples()
        return training_examples

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
        file_path = Path(self.runner.config.model_dir) / INPUT_EXAMPLES_FILE_NAME
        if not file_path.is_file():
            return False
        examples = []
        with file_path.open("r", encoding="utf8") as f:
            for line in f:
                ex = ujson.loads(line)
                examples.append(ex)
        self.all_examples = examples
        self.skip_first_self_play = True
        return str(file_path)

    def save_training_examples(self):
        model_dir = Path(self.runner.config.model_dir)
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)
        file_path = model_dir / INPUT_EXAMPLES_FILE_NAME
        with file_path.open("w", encoding="utf-8") as f:
            for line in self.all_examples:
                f.write(ujson.dumps(line, escape_forward_slashes=False) + "\n")
        return str(file_path)
