import os
import time
import random
import numpy
import math
import sys

from alpha_zero_general.pytorch_classification.utils import Bar, AverageMeter
from alpha_zero_general.NeuralNet import NeuralNet
from mathzero.model.math_estimator import math_estimator
from mathzero.environment_state import MathEnvironmentState


class NetConfig:
    def __init__(self, lr=0.001, dropout=0.2, epochs=10, batch_size=256):
        self.lr = lr
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size


class MathModel(NeuralNet):
    def __init__(self, game):
        import tensorflow as tf

        self.action_size = game.get_agent_actions_count()

        self.args = NetConfig()
        # Feature columns describe how to use the input.
        self.token_value_feature = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key="token_values", hash_bucket_size=12
            ),
            dimension=3,
        )
        self.feature_tokens_type = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key="token_types", hash_bucket_size=12, dtype=tf.int16
            ),
            dimension=4,
        )

        self.feature_node_count = tf.feature_column.numeric_column(key="node_count")
        self.feature_problem_type = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(
                key="problem_type", hash_bucket_size=3, dtype=tf.int16
            ),
            dimension=2,
        )
        self.feature_columns = [
            self.feature_problem_type,
            self.feature_tokens_type,
            self.feature_node_count,
            self.token_value_feature,
        ]

        self.nnet = tf.estimator.Estimator(
            model_fn=math_estimator,
            model_dir="training/embedding_one",
            params={
                "feature_columns": self.feature_columns,
                "action_size": self.action_size,
                "learning_rate": 0.1,
                "hidden_units": [8, 2],
            },
        )

    def train(self, examples):
        """
        examples: list of examples, each example is of form (env_state, pi, v)
        """
        import tensorflow as tf

        total_batches = int(len(examples) / self.args.batch_size)
        if total_batches == 0:
            return False

        print(
            "Training neural net for ({}) epochs with ({}) examples...".format(
                self.args.epochs, len(examples)
            )
        )

        def train_input_fn(examples, outputs, batch_size):
            dataset = tf.data.Dataset.from_tensor_slices(examples, outputs)
            assert examples.shape[0] == outputs.shape[0]
            dataset = dataset.shuffle(1000).batch(batch_size)
            return dataset

        self.classifier.train(
            input_fn=lambda: train_input_fn(train_x, train_y, self.args.batch_size),
            steps=self.args.epochs,
        )

        # for epoch in range(self.args.epochs):
        #     print("EPOCH ::: " + str(epoch + 1))
        #     data_time = AverageMeter()
        #     batch_time = AverageMeter()
        #     pi_losses = AverageMeter()
        #     v_losses = AverageMeter()
        #     end = time.time()

        #     bar = Bar("Training Net", max=int(len(examples) / self.args.batch_size))
        #     batch_idx = 0

        #     # self.session.run(tf.local_variables_initializer())
        #     while batch_idx < total_batches:
        #         sample_ids = numpy.random.randint(
        #             len(examples), size=self.args.batch_size
        #         )
        #         boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))

        #         # predict and compute gradient and do SGD step
        #         input_dict = {
        #             self.nnet.input_boards: boards,
        #             self.nnet.target_pis: pis,
        #             self.nnet.target_vs: vs,
        #             self.nnet.dropout: self.args.dropout,
        #             self.nnet.isTraining: True,
        #         }

        #         # measure data loading time
        #         data_time.update(time.time() - end)

        #         # record loss
        #         self.session.run(self.nnet.train_step, feed_dict=input_dict)
        #         pi_loss, v_loss = self.session.run(
        #             [self.nnet.loss_pi, self.nnet.loss_v], feed_dict=input_dict
        #         )
        #         pi_losses.update(pi_loss, len(boards))
        #         v_losses.update(v_loss, len(boards))

        #         # measure elapsed time
        #         batch_time.update(time.time() - end)
        #         end = time.time()
        #         batch_idx += 1

        #         # plot progress
        #         bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}".format(
        #             batch=batch_idx,
        #             size=int(len(examples) / self.args.batch_size),
        #             data=data_time.avg,
        #             bt=batch_time.avg,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             lpi=pi_losses.avg,
        #             lv=v_losses.avg,
        #         )
        #         bar.next()
        #     bar.finish()
        return True

    def predict(self, env_state: MathEnvironmentState):
        """
        env_state: numpy array with env_state
        """
        import tensorflow as tf

        start = time.time()
        predictions = self.nnet.predict(
            input_fn=lambda: predict_fn(env_state, batch_size=self.args.batch_size)
        )
        prediction = next(predictions)
        # print("predict : {0:03f}".format(time.time() - start))
        return prediction["out_policy"], prediction["out_value"][0]


def predict_fn(input_env_state: MathEnvironmentState, batch_size):
    import tensorflow as tf

    input_features = input_env_state.to_input_features()
    dataset = tf.data.Dataset.from_tensor_slices(input_features)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset
