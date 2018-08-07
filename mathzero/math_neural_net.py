import tensorflow as tf
import os
import time
import random
import numpy
import math
import sys

from alpha_zero_general.pytorch_classification.utils import Bar, AverageMeter
from alpha_zero_general.NeuralNet import NeuralNet
from mathzero.math_model import MathModel


class NetConfig:
    def __init__(
        self, lr=0.001, dropout=0.3, epochs=10, batch_size=256, num_channels=512
    ):
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_channels = num_channels


class MathNeuralNet(NeuralNet):
    def __init__(self, game):
        self.args = NetConfig()
        self.nnet = MathModel(game, self.args)
        self.board_x, self.board_y = game.getAgentStateSize()
        self.action_size = game.getActionSize()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(
            graph=self.nnet.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )
        self.saver = None
        with tf.Session() as temp_sess:
            temp_sess.run(tf.global_variables_initializer())
        self.sess.run(
            tf.variables_initializer(self.nnet.graph.get_collection("variables"))
        )

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """

        total_batches = int(len(examples) / self.args.batch_size)
        if total_batches == 0:
            return False

        print(
            "Training neural net for ({}) epochs with ({}) examples...".format(
                self.args.epochs, len(examples)
            )
        )

        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar("Training Net", max=int(len(examples) / self.args.batch_size))
            batch_idx = 0

            # self.sess.run(tf.local_variables_initializer())
            while batch_idx < total_batches:
                sample_ids = numpy.random.randint(
                    len(examples), size=self.args.batch_size
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

                # predict and compute gradient and do SGD step
                input_dict = {
                    self.nnet.input_boards: boards,
                    self.nnet.target_pis: pis,
                    self.nnet.target_vs: vs,
                    self.nnet.dropout: self.args.dropout,
                    self.nnet.isTraining: True,
                }

                # measure data loading time
                data_time.update(time.time() - end)

                # record loss
                self.sess.run(self.nnet.train_step, feed_dict=input_dict)
                pi_loss, v_loss = self.sess.run(
                    [self.nnet.loss_pi, self.nnet.loss_v], feed_dict=input_dict
                )
                pi_losses.update(pi_loss, len(boards))
                v_losses.update(v_loss, len(boards))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}".format(
                    batch=batch_idx,
                    size=int(len(examples) / self.args.batch_size),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()
            bar.finish()
            return True

    def predict(self, board):
        """
        board: numpy array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[numpy.newaxis, :, :]

        # run
        prob, v = self.sess.run(
            [self.nnet.prob, self.nnet.v],
            feed_dict={
                self.nnet.input_boards: board,
                self.nnet.dropout: 0,
                self.nnet.isTraining: False,
            },
        )

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return prob[0], v[0]

    def can_load_checkpoint(self, file_path) -> bool:
        meta = "{}.meta".format(file_path)
        return os.path.exists(meta)

    def save_checkpoint(self, file_path):
        dirname = os.path.dirname(file_path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        else:
            print("Checkpoint Directory exists for file: {}".format(file_path))
        if self.saver == None:
            self.saver = tf.train.Saver(self.nnet.graph.get_collection("variables"))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, file_path)

    def load_checkpoint(self, file_path):
        if not os.path.exists(file_path + ".meta"):
            raise Exception("No model in path {}".format(file_path))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, file_path)
