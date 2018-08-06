import sys
import types
from collections import deque
from .Arena import Arena
from .MCTS import MCTS
import numpy as np
from .pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import os
from multiprocessing import Pool, Array, Process, Queue, cpu_count
from .Game import Game

class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def has_examples(self) -> bool:
        return bool(len(self.training_examples_history) > 0)

    def __init__(self, runner, args=None):
        if args is None:
            args = dict()
        self.runner = runner
        self.game = runner.get_game()
        self.nnet = runner.get_nnet(self.game)
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.training_iterations = args.get("training_iterations", 50)
        self.self_play_iterations = args.get("self_play_iterations", 100)
        self.model_win_loss_ratio = args.get("model_win_loss_ratio", 0.6)
        self.max_training_examples = args.get("max_training_examples", 200000)
        self.model_arena_iterations = args.get("model_arena_iterations", 30)
        self.checkpoint = args.get("checkpoint", "./training/")
        self.best_model_name = args.get("best_model_name", "best")
        self.save_examples_from_last_n_iterations = args.get(
            "save_examples_from_last_n_iterations", 20
        )
        self.training_examples_history = []  # history of examples from args.save_examples_from_last_n_iterations latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()
        best = self.get_best_model_filename()
        if self.can_load_model(best):
            print("Starting with best existing model: {}".format(best))
            self.nnet.load_checkpoint(best)
            self.load_training_examples(best)

    def learn(self):
        """
        Performs training_iterations iterations with self_play_iterations episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= model_win_loss_ratio fraction of games.
        """
        # Where to store the current checkpoint while learning
        temp_file_path = os.path.join(self.checkpoint, "temp.pth.tar")
        iterations = self.self_play_iterations

        for i in range(1, iterations + 1):
            print("------ITER {}------".format(i))
            training_examples = deque([], maxlen=self.max_training_examples)

            if i > 1 or not self.skip_first_self_play:
                current_episode = 0
                eps_time = AverageMeter()
                bar = Bar("Self Play", max=iterations)
                bar.suffix = "Playing first game..."
                bar.next()

                def update_episode_bar(self, episode, duration):
                    nonlocal current_episode, bar, eps_time, iterations
                    # bookkeeping + plot progress
                    eps_time.update(duration)
                    current_episode += 1
                    bar.suffix = "({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}".format(
                        eps=current_episode,
                        maxeps=iterations,
                        et=eps_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                    )
                    bar.next()

                episodes_with_args = []
                for i in range(1, self.self_play_iterations + 1):
                    episodes_with_args.append(dict(player=1 if i % 2 == 0 else -1))

                old_update = self.runner.episode_complete
                self.runner.episode_complete = types.MethodType(
                    update_episode_bar, self
                )
                training_examples += self.runner.execute_episodes(episodes_with_args)
                self.runner.episode_complete = old_update
                bar.finish()

                # save the iteration examples to the history
                self.training_examples_history.append(training_examples)

                if (
                    len(self.training_examples_history)
                    > self.save_examples_from_last_n_iterations
                ):
                    print(
                        "len(trainExamplesHistory) =",
                        len(self.training_examples_history),
                        " => remove the oldest trainExamples",
                    )
                    self.training_examples_history.pop(0)
                # backup history to a file
                # NB! the examples were collected using the model from the previous iteration, so (i-1)
                self.save_training_examples(i - 1)

            # shuffle examlpes before training
            trainExamples = []
            for e in self.training_examples_history:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(temp_file_path)
            self.pnet.load_checkpoint(temp_file_path)
            pmcts = MCTS(self.game, self.pnet, self.runner.config.cpuct, self.runner.config.num_mcts_sims)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.runner.config.cpuct, self.runner.config.num_mcts_sims)

            print("PITTING AGAINST SELF-PLAY VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.model_arena_iterations)

            print("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins > 0
                and float(nwins) / (pwins + nwins) < self.model_win_loss_ratio
            ):
                print("REJECTING NEW MODEL")
                self.nnet.load_checkpoint(temp_file_path)
            else:
                print("ACCEPTING NEW MODEL")
                checkpoint_file = os.path.join(
                    self.checkpoint, self.get_checkpoint_filename(i)
                )
                self.nnet.save_checkpoint(checkpoint_file)
                self.save_current_model("best")

    def get_checkpoint_filename(self, iteration):
        return "checkpoint_{}.pth.tar".format(iteration)

    def get_best_model_filename(self):
        return os.path.join(self.checkpoint, "{}.pth.tar".format(self.best_model_name))

    def can_load_model(self, model_name):
        meta = "{}.meta".format(model_name)
        return os.path.exists(meta)

    def save_training_examples(self, iteration):
        folder = self.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder, self.get_checkpoint_filename(iteration) + ".examples"
        )
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.training_examples_history)

    def save_current_model(self, name="best"):
        folder = self.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(self.checkpoint, "{}.pth.tar".format(name))
        self.nnet.save_checkpoint(filename)
        examples_file = "{}.examples".format(filename)
        with open(examples_file, "wb+") as f:
            Pickler(f).dump(self.training_examples_history)

    def load_training_examples(self, name=None):
        examplesFile = "{}.examples".format(name)
        if not os.path.isfile(examplesFile):
            return False
        with open(examplesFile, "rb") as f:
            self.training_examples_history = Unpickler(f).load()
        # examples based on the model were already collected (loaded)
        self.skip_first_self_play = True
        return True
