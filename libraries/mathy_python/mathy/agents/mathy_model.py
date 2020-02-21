from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from mathy.agents.base_config import BaseConfig
from mathy.agents.densenet import DenseNetStack
from mathy.core.expressions import MathTypeKeysMax
from mathy.state import (
    MathyInputsType,
    MathyWindowObservation,
    ObservationFeatureIndices,
)


class TimeDistributed(torch.nn.Module):
    """PyTorch TimeDistributed from:
    https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py#L17-L38"""

    def __init__(self, module: torch.nn.Module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x: torch.Tensor):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + " (\n"
        tmpstr += self.module.__repr__()
        tmpstr += ")"
        return tmpstr


class MathyModel(torch.nn.Module):
    def __init__(self, config: BaseConfig, predictions: int, **kwargs):
        super(MathyModel, self).__init__(**kwargs)
        self.predictions = predictions
        self.config = config
        self.width = config.max_sequence_length
        self.token_embedding = torch.nn.Embedding(
            MathTypeKeysMax, self.config.embedding_units
        )

        self.in_dense_units = (
            self.config.embedding_units + self.config.units + self.config.units + 1
        )
        # +1 for the value
        # +1 for the time
        # +2 for the problem type hashes
        self.concat_size = 4 if self.config.use_env_features else 1
        if self.config.use_env_features:
            self.time_dense = torch.nn.Linear(1, self.config.units)
            self.type_dense = torch.nn.Linear(2, self.config.units)
        self.in_dense = torch.nn.Linear(self.in_dense_units, self.width)
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.width, self.config.embedding_units),
            torch.nn.ReLU(),
            torch.nn.LayerNorm((self.width, self.config.embedding_units)),
        )

    def forward(self, features: MathyInputsType):
        nodes = torch.Tensor(features[ObservationFeatureIndices.nodes]).long()
        values = torch.Tensor(features[ObservationFeatureIndices.values])
        type = torch.Tensor(features[ObservationFeatureIndices.type])
        time = torch.Tensor(features[ObservationFeatureIndices.time])
        batch_size = nodes.shape[0]
        sequence_length = nodes.shape[1]

        in_rnn_state_h = features[ObservationFeatureIndices.rnn_state_h]
        in_rnn_state_c = features[ObservationFeatureIndices.rnn_state_c]
        in_rnn_history_h = features[ObservationFeatureIndices.rnn_history_h]

        values = values.unsqueeze_(-1)
        query = self.token_embedding(nodes)
        type_tensor = self.type_dense(type)
        time_tensor = self.time_dense(time)
        # If not using env features, only concatenate the tokens and values
        env_inputs = [
            query,
            type_tensor,
            time_tensor,
        ]
        if self.config.use_node_values:
            env_inputs.insert(1, values)
        query = torch.cat(env_inputs, -1)

        # Input dense transforms
        query = self.in_dense(query)
        output = self.output_net(query)
        return output
