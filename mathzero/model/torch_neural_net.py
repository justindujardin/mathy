import os
import time
import random
import numpy
import math

from alpha_zero_general.pytorch_classification.utils import Bar, AverageMeter
from alpha_zero_general.NeuralNet import NeuralNet
from mathzero.model.torch_model import MathModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class NetConfig:
    def __init__(
        self,
        lr=0.001,
        dropout=0.3,
        epochs=10,
        batch_size=256,
        num_channels=512,
        cuda=torch.cuda.is_available(),
    ):
        self.lr = lr
        self.cuda = cuda
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_channels = num_channels


class MathNeuralNet(NeuralNet):
    def __init__(self, game, all_memory=False):
        self.args = NetConfig()
        self.nnet = MathModel(game, self.args)
        self.board_x, self.board_y = game.get_agent_state_size()
        self.action_size = game.get_agent_actions_count()

        if self.args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar("Training Net", max=int(len(examples) / self.args.batch_size))
            batch_idx = 0

            while batch_idx < int(len(examples) / self.args.batch_size):
                sample_ids = numpy.random.randint(
                    len(examples), size=self.args.batch_size
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(numpy.array(boards).astype(numpy.float64))
                target_pis = torch.FloatTensor(numpy.array(pis))
                target_vs = torch.FloatTensor(numpy.array(vs).astype(numpy.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = (
                        boards.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda(),
                    )
                boards, target_pis, target_vs = (
                    Variable(boards),
                    Variable(target_pis),
                    Variable(target_vs),
                )

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.data.item(), boards.size(0))
                v_losses.update(l_v.data.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

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
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(numpy.float64))
        if self.args.cuda:
            board = board.contiguous().cuda()
        with torch.no_grad():
            board = Variable(board)
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()
        pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def can_load_checkpoint(self, file_path) -> bool:
        return os.path.exists(file_path)

    def save_checkpoint(self, file_path: str):
        dirname = os.path.dirname(file_path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print("Creating checkpoint directory: {}".format(dirname))
        else:
            print("Checkpoint directory exists: {}".format(dirname))
        torch.save({"state_dict": self.nnet.state_dict()}, file_path)

    def load_checkpoint(self, file_path: str):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        if not os.path.exists(file_path):
            raise ValueError("No model in path {}".format(file_path))
        checkpoint = torch.load(file_path)
        self.nnet.load_state_dict(checkpoint["state_dict"])

