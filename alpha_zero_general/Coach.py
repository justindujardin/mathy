from collections import deque
from .Arena import Arena
from .MCTS import MCTS
import numpy as np
from .pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import os


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.num_mcts_sims = args.get("num_mcts_sims")
        self.cpuct = args.get("cpuct")
        self.training_iterations = args.get("training_iterations")
        self.self_play_iterations = args.get("self_play_iterations")
        self.temperature_threshold = args.get("temperature_threshold")
        self.model_win_loss_ratio = args.get("model_win_loss_ratio")
        self.max_training_examples = args.get("max_training_examples")
        self.model_arena_iterations = args.get("model_arena_iterations")
        self.checkpoint = args.get("checkpoint")
        self.best_model_name = args.get("best_model_name", "best")
        self.save_examples_from_last_n_iterations = args.get(
            "save_examples_from_last_n_iterations"
        )
        self.mcts = MCTS(
            self.game, self.nnet, cpuct=self.cpuct, num_mcts_sims=self.num_mcts_sims
        )

        self.best_model_name = args["best_model_name"]
        self.training_examples_history = []  # history of examples from args.save_examples_from_last_n_iterations latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()

        best = self.get_best_model_filename()
        if self.can_load_model(best):
            print("Starting with best existing model: {}".format(best))
            nnet.load_checkpoint(best)
            self.load_training_examples(best)
        else:
            print(
                "No existing checkpoint found, starting with a fresh model and self-play..."
            )

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game continues until getGameEnded returns a non-zero
        value, then the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < temperature_threshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.temperature_threshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action
            )
            # print("turn done, player is now: {}".format(self.curPlayer))
            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [
                    (x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)))
                    for x in trainExamples
                ]

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

        for i in range(1, self.training_iterations + 1):
            # bookkeeping
            print("------ITER " + str(i) + "------")
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iterationTrainExamples = deque([], maxlen=self.max_training_examples)

                eps_time = AverageMeter()
                bar = Bar("Self Play", max=self.self_play_iterations)
                end = time.time()

                for eps in range(self.self_play_iterations):
                    # reset search tree
                    self.mcts = MCTS(
                        self.game, self.nnet, self.cpuct, self.num_mcts_sims
                    )
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = "({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}".format(
                        eps=eps + 1,
                        maxeps=self.self_play_iterations,
                        et=eps_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                    )
                    bar.next()
                bar.finish()

                # save the iteration examples to the history
                self.training_examples_history.append(iterationTrainExamples)

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
            pmcts = MCTS(self.game, self.pnet, self.cpuct, self.num_mcts_sims)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.cpuct, self.num_mcts_sims)

            print("PITTING AGAINST PREVIOUS VERSION")
            arena = Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.model_arena_iterations)
            best_model_path = os.path.join(self.checkpoint, "best.pth.tar")
            has_best = os.path.isfile(best_model_path)

            print("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                has_best
                and pwins + nwins > 0
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
        examples_file = os.path.join(folder, "{}.examples".format(filename))
        with open(examples_file, "wb+") as f:
            Pickler(f).dump(self.training_examples_history)

    def load_training_examples(self, name=None):
        examplesFile = "{}.examples".format(name)
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input(
                "No examples found. Session will build new training examples. Continue? [y|N]"
            )
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.training_examples_history = Unpickler(f).load()
            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
