import json
import random
from collections import namedtuple
from typing import List, NamedTuple, Optional, Tuple

import numpy
import tensorflow as tf

from .helpers import get_term_ex, get_terms
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

PROBLEM_TYPE_HASH_BUCKETS = 128

INPUT_EXAMPLES_FILE_NAME = "examples.jsonl"
TRAINING_SET_FILE_NAME = "training.jsonl"

NodeIntList = List[int]
NodeHintMaskIntList = List[int]
NodeMaskIntList = List[int]
ProblemTypeIntList = List[int]
RNNStatesFloatList = List[List[List[float]]]

WindowNodeIntList = List[NodeIntList]
WindowNodeMaskIntList = List[NodeMaskIntList]
WindowNodeHintMaskIntList = List[NodeHintMaskIntList]
WindowProblemTypeIntList = List[ProblemTypeIntList]
WindowRNNStatesFloatList = List[RNNStatesFloatList]

BatchNodeIntList = List[WindowNodeIntList]
BatchNodeMaskIntList = List[WindowNodeMaskIntList]
BatchNodeHintMaskIntList = List[WindowNodeHintMaskIntList]
BatchProblemTypeIntList = List[WindowProblemTypeIntList]
BatchRNNStatesFloatList = List[WindowRNNStatesFloatList]


class MathyObservation(NamedTuple):
    """A featurized observation from an environment state.
    - "nodes" tree nodes in the current environment state shape=[n,]
    - "mask" 0/1 mask where 0 indicates an invalid action shape=[n,]
    - "hints" 0/1 mask where 1 indicates a useful node for the environment shape=[n,]
    - "type" two column hash of problem environment type shape=[2,]
    - "rnn_state" n-step rnn state paris shape=[2*dimensions]
    """

    nodes: NodeIntList
    mask: NodeMaskIntList
    hints: NodeHintMaskIntList
    type: ProblemTypeIntList
    rnn_state: RNNStatesFloatList


class MathyWindowObservation(NamedTuple):
    """A featurized observation from an n-step sequence of environment states.
    - "nodes" n-step list of node sequences shape=[n, max(len(s))]
    - "mask" n-step list of node sequence masks shape=[n, max(len(s))]
    - "hints" 0/1 mask where 1 indicates a useful node for the environment
       of shape=[n, max(len(s))]
    - "type" n-step problem type hash shape=[n, 2]
    - "rnn_state" n-step rnn state paris shape=[n, 2*dimensions]
    """

    nodes: WindowNodeIntList
    mask: WindowNodeMaskIntList
    hints: WindowNodeHintMaskIntList
    type: WindowProblemTypeIntList
    rnn_state: WindowRNNStatesFloatList


class MathyBatchObservation(NamedTuple):
    """A batch of featurized observations from n-step sequences of environment states.

    - "nodes" Batch of n-step node sequences shape=[b, n, max(len(s))]
    - "mask" Batch of n-step node sequence masks shape=[b, n, max(len(s))]
    - "hints" Batch of n-step node sequence hint masks of shape=[n, max(len(s))]
    - "type" Batch of n-step problem hashes shape=[b, n, max(len(s))]
    - "rnn_state" Batch of n-step rnn state pairs shape=[b, n, 2*dimensions]
    """

    nodes: BatchNodeIntList
    mask: BatchNodeMaskIntList
    hints: BatchNodeHintMaskIntList
    type: BatchProblemTypeIntList
    rnn_state: BatchRNNStatesFloatList


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


def rnn_placeholder_state(rnn_size: int) -> RNNStatesFloatList:
    rnn_state: RNNStatesFloatList = [
        tf.zeros(shape=(1, rnn_size)),
        tf.zeros(shape=(1, rnn_size)),
    ]
    return rnn_state


def rnn_placeholder_states(rnn_size: int, num_states: int) -> WindowRNNStatesFloatList:
    rnn_state: WindowRNNStatesFloatList = [
        [tf.zeros(shape=(1, rnn_size)), tf.zeros(shape=(1, rnn_size))]
        for _ in range(num_states)
    ]
    return rnn_state


def observations_to_window(
    observations: List[MathyObservation]
) -> MathyWindowObservation:
    output = MathyWindowObservation(
        nodes=[], mask=[], hints=[], type=[], rnn_state=[[], []]
    )
    max_length: int = max([len(o.nodes) for o in observations])
    max_mask_length: int = max([len(o.mask) for o in observations])

    for obs in observations:
        output.nodes.append(pad_array(obs.nodes, max_length, MathTypeKeys["empty"]))
        output.mask.append(pad_array(obs.mask, max_mask_length, 0))
        output.type.append(obs.type)
        output.rnn_state[0].append(obs.rnn_state[0])
        output.rnn_state[1].append(obs.rnn_state[1])
    return output


def windows_to_batch(windows: List[MathyWindowObservation]) -> MathyBatchObservation:
    output = MathyBatchObservation(
        nodes=[], mask=[], hints=[], type=[], rnn_state=[[], []]
    )
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

    def problem_hash(self, one_hot: bool = False) -> ProblemTypeIntList:
        problem_hash_one = tf.strings.to_hash_bucket_strong(
            self.agent.problem_type, PROBLEM_TYPE_HASH_BUCKETS, [1337, 61059873]
        )
        problem_hash_two = tf.strings.to_hash_bucket_strong(
            self.agent.problem_type, PROBLEM_TYPE_HASH_BUCKETS, [7, -242315]
        )
        if one_hot:
            hash_tensor = tf.one_hot(
                indices=[problem_hash_one, problem_hash_two],
                depth=PROBLEM_TYPE_HASH_BUCKETS,
            )
        else:
            hash_tensor = tf.convert_to_tensor([problem_hash_one, problem_hash_two])
        return hash_tensor.numpy().tolist()

    def to_empty_observation(self, hash=None, rnn_size: int = 128) -> MathyObservation:
        # TODO: max actions hardcode
        num_actions = 6
        if hash is None:
            hash = self.problem_hash()
        mask = [0] * num_actions
        hints = [0] * num_actions
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]],
            mask=mask,
            hints=hints,
            type=hash,
            rnn_state=rnn_placeholder_state(rnn_size),
        )

    def to_observation(
        self,
        move_mask: NodeMaskIntList,
        hint_mask: Optional[NodeHintMaskIntList] = None,
        rnn_state: Optional[RNNStatesFloatList] = None,
    ) -> MathyObservation:
        if hint_mask is None:
            hint_mask = move_mask
        if rnn_state is None:
            rnn_state = rnn_placeholder_state(128)
        expression = self.parser.parse(self.agent.problem)
        nodes: List[MathExpression] = expression.toList()
        vectors: NodeIntList = [t.type_id for t in nodes]
        return MathyObservation(
            nodes=vectors,
            mask=move_mask,
            hints=[],
            type=self.problem_hash(),
            rnn_state=rnn_state,
        )

    def to_empty_window(self, samples: int, rnn_size: int) -> MathyWindowObservation:
        """Generate an empty window of observations that can be passed
        through the model"""
        hash = self.problem_hash()
        window = observations_to_window(
            [self.to_empty_observation(hash, rnn_size) for _ in range(samples)]
        )
        return window

    def to_empty_batch(self, window_size: int, rnn_size) -> MathyBatchObservation:
        """Generate an empty batch of observations that can be passed
        through the model """
        return windows_to_batch([self.to_empty_window(window_size, rnn_size)])
