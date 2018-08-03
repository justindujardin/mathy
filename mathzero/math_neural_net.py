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


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.args = NetConfig()
        self.nnet = MathModel(game, self.args)
        self.board_x, self.board_y = game.getAgentStateSize()
        self.action_size = game.getActionSize()

        # GPU percentage warns of failed allocations... Only one model per GPU? :( :(
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(graph=self.nnet.graph) # , config=tf.ConfigProto(gpu_options=gpu_options))
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

        print("Training neural net for ({}) epochs...".format(self.args.epochs))

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
            while batch_idx < int(len(examples) / self.args.batch_size):
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

    def save_checkpoint(self, filepath):
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    dirname
                )
            )
            os.mkdir(dirname)
        else:
            print("Checkpoint Directory exists for file: {}".format(filepath))
        if self.saver == None:
            self.saver = tf.train.Saver(self.nnet.graph.get_collection("variables"))
        with self.nnet.graph.as_default():
            self.saver.save(self.sess, filepath)

    def load_checkpoint(self, filepath):
        if not os.path.exists(filepath + ".meta"):
            raise Exception("No model in path {}".format(filepath))
        with self.nnet.graph.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, filepath)