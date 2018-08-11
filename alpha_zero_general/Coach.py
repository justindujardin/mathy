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
from multiprocessing import Pool, Array, Process, Queue, cpu_count, Value
from .Game import Game


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
        self.checkpoint = args.get("checkpoint", "./training/")
        self.best_model_name = args.get("best_model_name", "best")
        self.all_examples = []  # history of examples from args.save_examples_from_last_n_iterations latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()
        best = self.get_best_model_filename()
        if self.can_load_model(best):
            print("Loading examples from best model training: {}".format(best))
            self.load_training_examples(best)

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
            print("------ITER {}------".format(i))
            # Run self-play episodes
            self.run_self_play(i, iterations)
            # game = self.runner.get_game()
            # Train the network with the gathered examples from self-play
            self.run_network_training(i)
            # new_net = self.run_network_training(i)
            # # if there is not enough data for training, None is returned
            # if new_net == False:
            #     continue

            # pmcts = MCTS(
            #     game,
            #     new_net,
            #     self.runner.config.cpuct,
            #     self.runner.config.num_mcts_sims,
            # )

            # nmcts = MCTS(
            #     game,
            #     new_net,
            #     self.runner.config.cpuct,
            #     self.runner.config.num_mcts_sims,
            # )

            # print("PITTING AGAINST SELF-PLAY VERSION")
            # arena = Arena(
            #     lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #     lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
            #     game,
            # )
            # pwins, nwins, draws = arena.playGames(self.model_arena_iterations)

            # print("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            # if (pwins == 0 and nwins == 0) or (
            #     pwins + nwins > 0
            #     and float(nwins) / (pwins + nwins) < self.model_win_loss_ratio
            # ):
            #     print("REJECTING NEW MODEL")
            # else:
            #     print("ACCEPTING NEW MODEL")
            #     self.save_model(new_net, "best")

    def run_self_play(self, iteration, num_episodes):
        if iteration < 1 and self.skip_first_self_play:
            return []
        eps_time = AverageMeter()
        bar = Bar("Self Play", max=num_episodes)
        bar.suffix = "Playing first game..."
        current_episode = 0

        def update_episode_bar(self, episode, duration):
            nonlocal current_episode, bar, eps_time, num_episodes
            # bookkeeping + plot progress
            eps_time.update(duration)
            current_episode += 1
            bar.suffix = "({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}".format(
                eps=current_episode,
                maxeps=num_episodes,
                et=eps_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
            )
            bar.next()

        episodes_with_args = []
        for j in range(1, num_episodes + 1):
            episodes_with_args.append(
                dict(
                    player=1 if j % 2 == 0 else -1, model=self.get_best_model_filename()
                )
            )

        old_update = self.runner.episode_complete
        self.runner.episode_complete = types.MethodType(update_episode_bar, self)
        training_examples = deque(
            self.runner.execute_episodes(episodes_with_args),
            maxlen=self.max_training_examples,
        )
        self.runner.episode_complete = old_update
        bar.finish()
        # NB! the examples were collected using the model from the previous iteration, so (iteration-1)
        # self.save_training_examples(iteration - 1)
        self.all_examples.extend(training_examples)
        return training_examples

    def run_network_training(self, iteration):
        """
        Train the current best model with the gathered training examples and return
        a tuple of (best, new) where best is the existing best trained model (or a blank
        one) and new is the model that was just trained.
        """
        best = self.get_best_model_filename()
        train_examples = self.all_examples
        # print(train_examples)
        print("Training with {} examples".format(len(train_examples)))
        return self.runner.train(iteration, train_examples, best)

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
            Pickler(f).dump(self.all_examples)

    def save_model(self, nnet, name="best"):
        folder = self.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(self.checkpoint, "{}.pth.tar".format(name))
        nnet.save_checkpoint(filename)
        examples_file = "{}.examples".format(filename)
        with open(examples_file, "wb+") as f:
            Pickler(f).dump(self.all_examples)

    def load_training_examples(self, name=None):
        examplesFile = "{}.examples".format(name)
        if not os.path.isfile(examplesFile):
            return False
        with open(examplesFile, "rb") as f:
            self.all_examples = Unpickler(f).load()
        # examples based on the model were already collected (loaded)
        self.skip_first_self_play = True
        return True
