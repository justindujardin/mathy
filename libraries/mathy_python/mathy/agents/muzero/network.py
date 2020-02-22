from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple

import torch
import torch.nn.functional as F

from .config import MuZeroConfig
from ...core.expressions import MathTypeKeysMax
from ...state import (
    MathyInputsType,
    MathyWindowObservation,
    ObservationFeatureIndices,
)
from .types import Action


PolicyLogitsDict = Dict[Action, float]


class NetworkOutput(NamedTuple):
    value: float
    reward: float
    policy_logits: PolicyLogitsDict
    hidden_state: List[float]


class Network(torch.nn.Module):
    def __init__(self, config: MuZeroConfig, actions_per_node: int):
        super(Network, self).__init__()
        self.config = config
        self.actions_per_node = actions_per_node
        self.hidden_size = 32
        self.units = 128
        self.representation_net = RepresentationModel(config=config)
        self.dynamics_net = DynamicsModel(config=config, hidden_size=self.hidden_size)
        self.prediction_net = PredictionModel(
            config=config,
            hidden_size=self.hidden_size,
            actions_per_node=actions_per_node,
        )

    def initial_inference(self, input_features: MathyInputsType) -> NetworkOutput:
        """representation + prediction function"""
        hidden_state = self.representation_net(input_features)
        value, reward, policy = self.prediction_net(hidden_state)
        policy_dict = self.to_policy_dict(policy)
        return NetworkOutput(value, reward, policy_dict, hidden_state)

    def recurrent_inference(
        self, hidden_state: List[float], action: int
    ) -> NetworkOutput:
        """dynamics + prediction function"""
        new_hidden_state = self.dynamics_net(hidden_state)
        value, reward, policy = self.prediction_net(new_hidden_state)
        policy_dict = self.to_policy_dict(policy)
        return NetworkOutput(value, reward, policy_dict, new_hidden_state)

    def to_policy_dict(self, policy: torch.Tensor) -> PolicyLogitsDict:
        """Convert Logits tensor into a dictionary of { action: probability }"""
        p_list = policy.squeeze_().tolist()
        out_dict: PolicyLogitsDict = dict()
        for i, p in enumerate(p_list):
            out_dict[Action(i)] = p
        return out_dict

    def get_weights(self):
        # Returns the weights of this network.
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0


class RepresentationModel(torch.nn.Module):
    """Build a hidden state from an environment observation"""

    def __init__(self, config: MuZeroConfig, **kwargs):
        super(RepresentationModel, self).__init__(**kwargs)
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
        self.time_dense = torch.nn.Linear(1, self.config.units)
        self.type_dense = torch.nn.Linear(2, self.config.units)
        self.output_net = torch.nn.Sequential(
            torch.nn.Linear(self.in_dense_units, self.config.embedding_units),
            torch.nn.ReLU(),
            torch.nn.LayerNorm((self.width, self.config.embedding_units)),
        )

    def forward(self, features: MathyInputsType) -> torch.Tensor:
        nodes = torch.Tensor(features[ObservationFeatureIndices.nodes]).long()
        values = torch.Tensor(features[ObservationFeatureIndices.values])
        type = torch.Tensor(features[ObservationFeatureIndices.type])
        time = torch.Tensor(features[ObservationFeatureIndices.time])
        batch_size = nodes.shape[0]
        sequence_length = nodes.shape[1]
        values = values.unsqueeze_(-1)
        query = self.token_embedding(nodes)
        type_tensor = self.type_dense(type)
        time_tensor = self.time_dense(time)
        query = torch.cat([query, values, type_tensor, time_tensor], -1)
        output = self.output_net(query)
        return output


class DynamicsModel(torch.nn.Module):
    """Predict the next hidden state from the current"""

    def __init__(self, *, hidden_size: int, config: MuZeroConfig):
        super(DynamicsModel, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.dynamics_net = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.config.units),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.units, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_size),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        new_hidden_state = self.dynamics_net(hidden_state)
        return new_hidden_state


class PredictionModel(torch.nn.Module):
    """Prediction value, reward, policy from a hidden state"""

    def __init__(
        self, *, hidden_size: int, actions_per_node: int, config: MuZeroConfig
    ):
        super(PredictionModel, self).__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.actions_per_node = actions_per_node
        self.policy_size = self.actions_per_node * self.config.max_sequence_length
        # TODO: add reasonable heads for each prediction
        self.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.policy_size),
            torch.nn.LayerNorm(self.policy_size),
        )
        self.value_net = torch.nn.Linear(self.hidden_size, 1)
        self.reward_net = torch.nn.Linear(self.hidden_size, 1)

    def forward(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_hidden = torch.mean(hidden_state, -1)
        value = self.value_net(flat_hidden)
        reward = self.reward_net(flat_hidden)
        policy = self.policy_net(flat_hidden)
        return value, reward, policy

