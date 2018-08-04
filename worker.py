#!/usr/bin/env python
# coding: utf8
"""Construct a game instance and model for training or evaluation"""
import plac
from collections import deque
from alpha_zero_general.Arena import Arena
from alpha_zero_general.MCTS import MCTS
import numpy as np
from alpha_zero_general.pytorch_classification.utils import Bar, AverageMeter
import time, os, sys
from pickle import Pickler, Unpickler
from random import shuffle
import os
import concurrent.futures
import threading
import multiprocessing
from mathzero.math_game import MathGame
from mathzero.math_neural_net import NNetWrapper as nn
from mathzero.core.expressions import ConstantExpression
from mathzero.core.parser import ExpressionParser

eps = 250
temp = int(eps * 0.5)
arena = min(int(eps * 0.3), 30)

args = {
    "training_iterations": 1000,
    "self_play_iterations": eps,
    "temperature_threshold": temp,
    "model_win_loss_ratio": 0.6,
    "max_training_examples": 200000,
    "num_mcts_sims": 20,
    "model_arena_iterations": arena,
    "cpuct": 1,
    "checkpoint": "./training/temp/",
    "best_model_name": "best",
    "save_examples_from_last_n_iterations": 20,
}


if __name__ == "__main__":
    g = MathGame()
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()


def executeEpisode(packed_args):
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
    [game, nnet, player, num_mcts_sims, temperature_threshold, cpuct] = packed_args
    # print(
    #     "[{}] Starting episode {}....".format(threading.current_thread().name, episode)
    # )
    episode_examples = []
    board = game.getInitBoard()
    current_player = player
    move_count = 0
    mcts = MCTS(game, nnet, cpuct, num_mcts_sims)
    while True:
        move_count += 1
        canonical_state = game.getCanonicalForm(board, current_player)
        temp = int(move_count < temperature_threshold)

        pi = mcts.getActionProb(canonical_state, temp=temp)
        sym = game.getSymmetries(canonical_state, pi)
        for b, p in sym:
            episode_examples.append([b, current_player, p, None])
        action = np.random.choice(len(pi), p=pi)
        board, current_player = game.getNextState(board, current_player, action)
        r = game.getGameEnded(board, current_player)

        if r != 0:
            return [
                (x[0], x[2], r * ((-1) ** (x[1] != current_player)))
                for x in episode_examples
            ]


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.num_mcts_sims = args.get("num_mcts_sims", 15)
        self.cpuct = args.get("cpuct", 1.0)
        self.training_iterations = args.get("training_iterations", 50)
        self.self_play_iterations = args.get("self_play_iterations", 100)
        self.temperature_threshold = args.get("temperature_threshold", 15)
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
            nnet.load_checkpoint(best)
            self.load_training_examples(best)
            self.skip_first_self_play = True
        else:
            print(
                "No existing checkpoint found, starting with a fresh model and self-play..."
            )

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
            print("------ITER " + str(i) + "------")
            training_examples = deque([], maxlen=self.max_training_examples)
            if i > 1 or not self.skip_first_self_play:
                bar = Bar("Self Play", max=self.self_play_iterations)
                eps_time = AverageMeter()
                end = time.time()
                bar.suffix = "Playing first game..."
                bar.next()
                args = [
                    [
                        self.game,
                        self.nnet,
                        1 if i % 2 == 0 else -1,
                        self.num_mcts_sims,
                        self.temperature_threshold,
                        self.cpuct,
                    ]
                    for i in range(1, self.self_play_iterations + 1)
                ]
                eps = 0
                for packed_args in args:
                    examples = executeEpisode(packed_args)
                    training_examples += examples
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
                    eps += 1
                    bar.next()
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
            pmcts = MCTS(self.game, self.pnet, self.cpuct, self.num_mcts_sims)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.cpuct, self.num_mcts_sims)

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


@plac.annotations(
    model_path=("Path to loadable neural network model wrapper", "positional", None, str),
    out_loc=("Path to output folder writing tensors and labels data", "positional", None, str),
    name=("Human readable name for tsv file and vectors tensor", "positional", None, str),
)
def main(vectors_loc, out_loc, name="spaCy_vectors"):
    # A tab-separated file that contains information about the vectors for visualization
    #
    # Learn more: https://www.tensorflow.org/programmers_guide/embedding#metadata
    meta_file = "{}_labels.tsv".format(name)
    out_meta_file = path.join(out_loc, meta_file)

    print('Loading spaCy vectors model: {}'.format(vectors_loc))
    model = spacy.load(vectors_loc)

    print('Finding lexemes with vectors attached: {}'.format(vectors_loc))
    voacb_strings = [
        w for w in tqdm.tqdm(model.vocab.strings, total=len(model.vocab.strings), leave=False)
        if model.vocab.has_vector(w)
    ]
    vector_count = len(voacb_strings)

    print('Building Projector labels for {} vectors: {}'.format(vector_count, out_meta_file))
    vector_dimensions = model.vocab.vectors.shape[1]
    tf_vectors_variable = numpy.zeros((vector_count, vector_dimensions), dtype=numpy.float32)

    # Write a tab-separated file that contains information about the vectors for visualization
    #
    # Reference: https://www.tensorflow.org/programmers_guide/embedding#metadata
    with open(out_meta_file, 'wb') as file_metadata:
        # Define columns in the first row
        file_metadata.write("Text\tFrequency\n".encode('utf-8'))
        # Write out a row for each vector that we add to the tensorflow variable we created
        vec_index = 0

        for text in tqdm.tqdm(voacb_strings, total=len(voacb_strings), leave=False):
            # https://github.com/tensorflow/tensorflow/issues/9094
            text = '<Space>' if text.lstrip() == '' else text
            lex = model.vocab[text]

            # Store vector data and metadata
            tf_vectors_variable[vec_index] = numpy.float64(model.vocab.get_vector(text))
            file_metadata.write("{}\t{}\n".format(text, math.exp(lex.prob) * len(voacb_strings)).encode('utf-8'))
            vec_index += 1

    # Write out "[name]_tensors.bytes" file for standalone embeddings projector to load
    tensor_path = '{}_tensors.bytes'.format(name)
    tf_vectors_variable.tofile(path.join(out_loc, tensor_path))

    print('Done.')
    print('Add the following entry to "oss_data/oss_demo_projector_config.json"')
    print(json.dumps({
        "tensorName": name,
        "tensorShape": [vector_count, vector_dimensions],
        "tensorPath": 'oss_data/{}'.format(tensor_path),
        "metadataPath": 'oss_data/{}'.format(meta_file)
    }, indent=2))


if __name__ == '__main__':
    plac.call(main)