import json
import random
from collections import namedtuple
from typing import List, NamedTuple, Optional, Tuple

import numpy
import tensorflow as tf

from .core.expressions import MathExpression, MathTypeKeys, MathTypeKeysMax
from .core.parser import ExpressionParser
from .core.tokenizer import TokenEOF
from .core.tree import STOP
from .features import (
    FEATURE_BWD_VECTORS,
    FEATURE_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_RULE,
    FEATURE_MOVE_COUNTER,
    FEATURE_MOVE_MASK,
    FEATURE_MOVES_REMAINING,
    FEATURE_NODE_COUNT,
    FEATURE_PROBLEM_TYPE,
    pad_array,
)
from .game_modes import (
    MODE_ARITHMETIC,
    MODE_SIMPLIFY_POLYNOMIAL,
    MODE_SOLVE_FOR_VARIABLE,
)

PLAYER_ID_OFFSET = 0
MOVE_COUNT_OFFSET = 1
GAME_MODE_OFFSET = 2

INPUT_EXAMPLES_FILE_NAME = "examples.jsonl"
TRAINING_SET_FILE_NAME = "training.jsonl"

NodeIntList = List[int]
NodeMaskIntList = List[int]
ProblemTypeIntList = List[int]

WindowNodeIntList = List[NodeIntList]
WindowNodeMaskIntList = List[NodeMaskIntList]
WindowProblemTypeIntList = List[ProblemTypeIntList]

BatchNodeIntList = List[WindowNodeIntList]
BatchNodeMaskIntList = List[WindowNodeMaskIntList]
BatchProblemTypeIntList = List[WindowProblemTypeIntList]


class MathyObservation(NamedTuple):
    """A featurized observation from an environment state.
    - "nodes" tree nodes in the current environment state shape=[n,]
    - "mask" 0/1 mask where 0 indicates an invalid action shape=[n,]
    - "type" two column hash of problem environment type shape=[2,]
    """

    nodes: NodeIntList
    mask: NodeMaskIntList
    type: ProblemTypeIntList


class MathyWindowObservation(NamedTuple):
    """A featurized observation from an n-step sequence of environment states.
    - "nodes" n-step list of node sequences shape=[n, max(len(s))]
    - "mask" n-step list of node sequence masks shape=[n, max(len(s))]
    - "type" n-step problem type hash shape=[n, 2]
    """

    nodes: WindowNodeIntList
    mask: WindowNodeMaskIntList
    type: WindowProblemTypeIntList


class MathyBatchObservation(NamedTuple):
    """A batch of featurized observations from n-step sequences of environment states.

    - "nodes" Batch of n-step node sequences shape=[b, n, max(len(s))]
    - "mask" Batch of n-step node sequence masks shape=[b, n, max(len(s))]
    - "type" Batch of n-step problem hashes shape=[b, n, max(len(s))]
    """

    nodes: BatchNodeIntList
    mask: BatchNodeMaskIntList
    type: BatchProblemTypeIntList


class MathyEnvTimeStep(NamedTuple):
    """Capture summarized environment state for a previous timestep so the
    agent can use context from its history when making new predictions.
    - "raw" is the input text at the timestep
    - "focus" is the index of the node that is acted on
    - "action" is the action taken
    """

    raw: str
    focus: int
    action: int


def observations_to_window(
    observations: List[MathyObservation]
) -> MathyWindowObservation:
    output = MathyWindowObservation([], [], [])
    max_length: int = max([len(o.nodes) for o in observations])
    max_mask_length: int = max([len(o.mask) for o in observations])
    for obs in observations:
        output.nodes.append(pad_array(obs.nodes, max_length, MathTypeKeys["empty"]))
        output.mask.append(pad_array(obs.mask, max_mask_length, 0))
        output.type.append(obs.type)
    return output


def windows_to_batch(windows: List[MathyWindowObservation]) -> MathyBatchObservation:
    output = MathyBatchObservation([], [], [])
    max_length = 0
    max_mask_length = 0
    for window in windows:
        window_max = max([len(ts) for ts in window.nodes])
        window_mask_max = max([len(ts) for ts in window.mask])
        max_length = max([max_length, window_max])
        max_mask_length = max([max_mask_length, window_mask_max])

    for window in windows:
        window_nodes = []
        for nodes in window.nodes:
            window_nodes.append(pad_array(nodes, max_length, MathTypeKeys["empty"]))
        window_mask = []
        for mask in window.mask:
            window_mask.append(pad_array(mask, max_mask_length, 0))
        output.nodes.append(window_nodes)
        output.mask.append(window_mask)
        output.type.append(window.type)
    return output


class MathAgentState:
    moves_remaining: int
    problem: str
    problem_type: str
    reward: float
    history: List[MathyEnvTimeStep]
    focus_index: int
    last_action: int

    def __init__(
        self,
        moves_remaining,
        problem,
        problem_type,
        reward=0.0,
        history=None,
        focus_index=0,
        last_action=None,
    ):
        self.moves_remaining = moves_remaining
        self.focus_index = focus_index  # TODO: remove this, no longer used
        self.problem = problem
        self.last_action = last_action
        self.reward = reward
        self.problem_type = problem_type
        self.history = (
            history[:] if history is not None else [MathyEnvTimeStep(problem, -1, -1)]
        )

    @classmethod
    def copy(cls, from_state):
        return MathAgentState(
            moves_remaining=from_state.moves_remaining,
            problem=from_state.problem,
            reward=from_state.reward,
            problem_type=from_state.problem_type,
            history=from_state.history,
            focus_index=from_state.focus_index,
            last_action=from_state.last_action,
        )


class MathyEnvState(object):
    """Class for holding environment state and extracting features
    to be passed to the policy/value neural network.

    Mutating operations all return a copy of the environment adapter
    with its own state.

    This allocation strategy requires more memory but removes a class
    of potential issues around unintentional sharing of data and mutation
    by two different sources.
    """

    agent: MathAgentState
    parser: ExpressionParser
    max_moves: int

    def __init__(
        self,
        state: Optional["MathyEnvState"] = None,
        problem: str = None,
        max_moves: int = 10,
        problem_type: str = "mathy.unknown",
    ):
        self.parser = ExpressionParser()
        self.max_moves = max_moves
        if problem is not None:
            self.agent = MathAgentState(max_moves, problem, problem_type)
        elif state is not None:
            self.max_moves = state.max_moves
            self.agent = MathAgentState.copy(state.agent)
        else:
            raise ValueError("either state or a problem must be provided")
        # Ensure a string made it into the problem slot
        assert isinstance(self.agent.problem, str)

    @classmethod
    def copy(cls, from_state):
        return MathyEnvState(state=from_state)

    def clone(self):
        return MathyEnvState(state=self)

    def encode_player(
        self, problem: str, action: int, focus_index: int, moves_remaining: int
    ):
        """Encode a player's state into the env_state, and return the env_state"""
        out_state = MathyEnvState.copy(self)
        agent = out_state.agent
        agent.history.append(MathyEnvTimeStep(problem, focus_index, action))
        agent.problem = problem
        agent.action = action
        agent.moves_remaining = moves_remaining
        agent.focus_index = focus_index
        return out_state

    def problem_hash(self) -> ProblemTypeIntList:
        max_problem_buckets = 128
        problem_hash_one = tf.strings.to_hash_bucket_strong(
            self.agent.problem_type, max_problem_buckets, [1337, 61059873]
        )
        problem_hash_two = tf.strings.to_hash_bucket_strong(
            self.agent.problem_type, max_problem_buckets, [7, -242315]
        )
        hash_tensor = tf.one_hot(
            indices=[problem_hash_one, problem_hash_two], depth=max_problem_buckets
        )
        return hash_tensor.numpy().tolist()

    def to_empty_observation(self, hash=None) -> MathyObservation:
        # TODO: max actions hardcode
        num_actions = 6
        if hash is None:
            hash = self.problem_hash()
        mask = [0] * num_actions
        return MathyObservation([MathTypeKeys["empty"]], mask, hash)

    def to_observation(self, move_mask: NodeMaskIntList) -> MathyObservation:
        expression = self.parser.parse(self.agent.problem)
        vectors: NodeIntList = [t.type_id for t in expression.toList()]
        return MathyObservation(vectors, move_mask, self.problem_hash())

    def to_empty_batch(self) -> MathyBatchObservation:
        """Generate an empty batch of observations that can be passed
        through the model """
        hash = self.problem_hash()
        window = observations_to_window(
            [
                self.to_empty_observation(hash),
                self.to_empty_observation(hash),
                self.to_empty_observation(hash),
            ]
        )
        return windows_to_batch([window])

    def to_input_features(self, move_mask: NodeMaskIntList, return_batch=False):
        """Output Dict of integer features that can be fed to the
        neural network for prediction.

        When return_batch is true, the outputs will have an array of features for each
        so the result can be directly passed to the predictor.
        """
        expression = self.parser.parse(self.agent.problem)
        # Generate context vectors for the current state's expression tree
        vectors = [[t.type_id] for t in expression.toList()]
        # # Start each sequence with the start token
        # vectors.insert(0, [TokenTypeSequenceStart])
        # # End it with a specific end token
        # vectors.append([TokenTypeSequenceEnd])

        # Encode the last action
        last_action = -1
        if len(self.agent.history) >= 1:
            last_action = self.agent.history[-1].action

        # After padding is done, generate reversed vectors for the bilstm
        vectors_reversed = vectors[:]
        vectors_reversed.reverse()

        def maybe_wrap(value):
            nonlocal return_batch
            return [value] if return_batch else value

        return {
            FEATURE_MOVES_REMAINING: maybe_wrap(self.agent.moves_remaining),
            FEATURE_MOVE_COUNTER: maybe_wrap(
                int(self.max_moves - self.agent.moves_remaining)
            ),
            FEATURE_LAST_RULE: maybe_wrap(last_action),
            FEATURE_NODE_COUNT: maybe_wrap(len(expression.toList())),
            FEATURE_PROBLEM_TYPE: maybe_wrap(self.agent.problem_type),
            FEATURE_FWD_VECTORS: maybe_wrap(vectors),
            FEATURE_BWD_VECTORS: maybe_wrap(vectors_reversed),
            FEATURE_MOVE_MASK: maybe_wrap(move_mask),
        }
