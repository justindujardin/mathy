from typing import List, NamedTuple, Optional, Dict

import tensorflow as tf
import numpy as np
from .core.expressions import ConstantExpression, MathExpression, MathTypeKeys
from .core.parser import ExpressionParser
from .util import pad_array

PROBLEM_TYPE_HASH_BUCKETS = 128

NodeIntList = List[int]
NodeValuesFloatList = List[float]
NodeMaskIntList = List[int]
ProblemTypeIntList = List[int]
ProblemTimeFloatList = List[float]
RNNStatesFloatList = List[List[List[float]]]

WindowNodeIntList = List[NodeIntList]
WindowNodeMaskIntList = List[NodeMaskIntList]
WindowNodeValuesFloatList = List[NodeValuesFloatList]
WindowProblemTypeIntList = List[ProblemTypeIntList]
WindowProblemTimeFloatList = List[ProblemTimeFloatList]
WindowRNNStatesFloatList = List[RNNStatesFloatList]

BatchNodeIntList = List[WindowNodeIntList]
BatchNodeMaskIntList = List[WindowNodeMaskIntList]
BatchNodeValuesFloatList = List[WindowNodeValuesFloatList]
BatchProblemTypeIntList = List[WindowProblemTypeIntList]
BatchProblemTimeFloatList = List[WindowProblemTimeFloatList]
BatchRNNStatesFloatList = List[WindowRNNStatesFloatList]


class MathyObservation(NamedTuple):
    """A featurized observation from an environment state.
    - "nodes" tree node types in the current environment state shape=[n,]
    - "mask" 0/1 mask where 0 indicates an invalid action shape=[n,]
    - "values" tree node value sequences, with non number indices set to 0.0
    - "type" two column hash of problem environment type shape=[2,]
    - "time" float value between 0.0-1.0 for how much time has passed shape=[1,]
    - "rnn_state" n-step rnn state paris shape=[2*dimensions]
    """

    nodes: NodeIntList
    mask: NodeMaskIntList
    values: NodeValuesFloatList
    type: ProblemTypeIntList
    time: ProblemTimeFloatList
    rnn_state: RNNStatesFloatList
    rnn_history: RNNStatesFloatList


class MathyWindowObservation(NamedTuple):
    """A featurized observation from an n-step sequence of environment states.
    - "nodes" n-step list of node sequences shape=[n, max(len(s))]
    - "mask" n-step list of node sequence masks shape=[n, max(len(s))]
    - "values" n-step list of value sequences, with non number indices set to 0.0
    - "type" n-step problem type hash shape=[n, 2]
    - "time" float value between 0.0-1.0 for how much time has passed shape=[n, 1]
    - "rnn_state" n-step rnn state paris shape=[n, 2*dimensions]
    """

    nodes: WindowNodeIntList
    mask: WindowNodeMaskIntList
    values: WindowNodeValuesFloatList
    type: WindowProblemTypeIntList
    time: WindowProblemTimeFloatList
    rnn_state: WindowRNNStatesFloatList
    rnn_history: WindowRNNStatesFloatList


class MathyBatchObservation(NamedTuple):
    """A batch of featurized observations from n-step sequences of environment states.

    - "nodes" Batch of n-step node sequences shape=[b, n, max(len(s))]
    - "mask" Batch of n-step node sequence masks shape=[b, n, max(len(s))]
    - "values" Batch of n-step list of value sequences, with non-number indices = 0.0
    - "type" Batch of n-step problem hashes shape=[b, n, max(len(s))]
    - "time" float value between 0.0-1.0 for how much time has passed shape=[b, n, 1]
    - "rnn_state" Batch of n-step rnn state pairs shape=[b, n, 2*dimensions]
    """

    nodes: BatchNodeIntList
    mask: BatchNodeMaskIntList
    values: BatchNodeValuesFloatList
    type: BatchProblemTypeIntList
    time: BatchProblemTimeFloatList
    rnn_state: BatchRNNStatesFloatList
    rnn_history: BatchRNNStatesFloatList


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


_cached_placeholder: Optional[RNNStatesFloatList] = None
_problem_hash_cache: Optional[Dict[str, List[int]]] = None


def rnn_placeholder_state(rnn_size: int) -> RNNStatesFloatList:
    """Create a placeholder state for the RNN hidden states in an observation. This
    is useful because we don't always know the RNN state when we initialize an
    observation. """
    global _cached_placeholder
    # Not in cache, or rnn_size differs
    if _cached_placeholder is None or len(_cached_placeholder[0][0]) != rnn_size:
        _cached_placeholder = [
            [[0.0] * rnn_size],
            [[0.0] * rnn_size],
        ]
    return _cached_placeholder


def observations_to_window(
    observations: List[MathyObservation],
) -> MathyWindowObservation:
    """Combine a sequence of observations into an observation window"""
    output = MathyWindowObservation(
        nodes=[],
        mask=[],
        values=[],
        type=[],
        time=[],
        rnn_state=[[], []],
        rnn_history=[[], []],
    )
    max_length: int = max([len(o.nodes) for o in observations])
    max_mask_length: int = max([len(o.mask) for o in observations])
    max_values_length: int = max([len(o.values) for o in observations])

    for obs in observations:
        output.nodes.append(pad_array(obs.nodes, max_length, MathTypeKeys["empty"]))
        output.mask.append(pad_array(obs.mask, max_mask_length, 0))
        output.values.append(pad_array(obs.values, max_values_length, 0.0))
        output.type.append(obs.type)
        output.time.append(obs.time)
        output.rnn_state[0].append(obs.rnn_state[0])
        output.rnn_state[1].append(obs.rnn_state[1])
        output.rnn_history[0].append(obs.rnn_history[0])
        output.rnn_history[1].append(obs.rnn_history[1])
    return output


def windows_to_batch(windows: List[MathyWindowObservation]) -> MathyBatchObservation:
    """Combine a sequence of window observations into a batched observation window"""
    output = MathyBatchObservation(
        nodes=[],
        mask=[],
        values=[],
        type=[],
        time=[],
        rnn_state=[[], []],
        rnn_history=[[], []],
    )
    max_length = 0
    max_mask_length = 0
    max_values_length = 0
    for window in windows:
        window_max = max([len(ts) for ts in window.nodes])
        window_mask_max = max([len(ts) for ts in window.mask])
        window_values_max = max([len(ts) for ts in window.values])
        max_length = max([max_length, window_max])
        max_mask_length = max([max_mask_length, window_mask_max])
        max_values_length = max([max_values_length, window_values_max])

    for window in windows:
        window_nodes = []
        for nodes in window.nodes:
            window_nodes.append(pad_array(nodes, max_length, MathTypeKeys["empty"]))
        window_mask = []
        for mask in window.mask:
            window_mask.append(pad_array(mask, max_mask_length, 0))
        window_values = []
        for value in window.values:
            window_values.append(pad_array(value, max_values_length, 0.0))
        output.nodes.append(window_nodes)
        output.mask.append(window_mask)
        output.values.append(window_values)
        output.type.append(window.type)
        output.time.append(window.time)
    return output


class MathyEnvState(object):
    """Class for holding environment state and extracting features
    to be passed to the policy/value neural network.

    Mutating operations all return a copy of the environment adapter
    with its own state.

    This allocation strategy requires more memory but removes a class
    of potential issues around unintentional sharing of data and mutation
    by two different sources.
    """

    agent: "MathyAgentState"
    parser: ExpressionParser
    max_moves: int
    num_rules: int

    def __init__(
        self,
        state: Optional["MathyEnvState"] = None,
        problem: str = None,
        max_moves: int = 10,
        num_rules: int = 0,
        problem_type: str = "mathy.unknown",
    ):
        self.parser = ExpressionParser()
        self.max_moves = max_moves
        self.num_rules = num_rules
        if problem is not None:
            self.agent = MathyAgentState(max_moves, problem, problem_type)
        elif state is not None:
            self.num_rules = state.num_rules
            self.max_moves = state.max_moves
            self.agent = MathyAgentState.copy(state.agent)
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
        global _problem_hash_cache
        if _problem_hash_cache is None:
            _problem_hash_cache = {}
        if self.agent.problem_type not in _problem_hash_cache:
            problem_hash_one = tf.strings.to_hash_bucket_strong(
                self.agent.problem_type, PROBLEM_TYPE_HASH_BUCKETS, [1337, 61059873]
            )
            problem_hash_two = tf.strings.to_hash_bucket_strong(
                self.agent.problem_type, PROBLEM_TYPE_HASH_BUCKETS, [7, -242315]
            )
            hash_tensor = tf.convert_to_tensor([problem_hash_one, problem_hash_two])
            _problem_hash_cache[self.agent.problem_type] = hash_tensor.numpy().tolist()
        return _problem_hash_cache[self.agent.problem_type]

    def to_start_observation(self, rnn_state: RNNStatesFloatList) -> MathyObservation:
        num_actions = 1 * self.num_rules
        hash = self.problem_hash()
        mask = [0] * num_actions
        values = [0.0] * num_actions
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]],
            mask=mask,
            values=values,
            type=hash,
            time=[0.0],
            rnn_state=rnn_state,
            rnn_history=rnn_placeholder_state(len(rnn_state[0][0])),
        )

    def to_empty_observation(self, hash=None, rnn_size: int = 128) -> MathyObservation:
        num_actions = 1 * self.num_rules
        if hash is None:
            hash = self.problem_hash()
        mask = [0] * num_actions
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]],
            mask=mask,
            values=[0.0],
            type=hash,
            time=[0.0],
            rnn_state=rnn_placeholder_state(rnn_size),
            rnn_history=rnn_placeholder_state(rnn_size),
        )

    def to_observation(
        self,
        move_mask: NodeMaskIntList,
        rnn_state: Optional[RNNStatesFloatList] = None,
        rnn_history: Optional[RNNStatesFloatList] = None,
    ) -> MathyObservation:
        if rnn_state is None:
            rnn_state = rnn_placeholder_state(128)
        if rnn_history is None:
            rnn_history = rnn_placeholder_state(128)
        expression = self.parser.parse(self.agent.problem)
        nodes: List[MathExpression] = expression.toList()
        vectors: NodeIntList = []
        values: NodeValuesFloatList = []
        for node in nodes:
            vectors.append(node.type_id)
            if isinstance(node, ConstantExpression):
                values.append(float(node.value))
            else:
                values.append(0.0)

        # Pass a 0-1 value indicating the relative episode time where 0.0 is
        # the episode start, and 1.0 is the episode end as indicated by the
        # maximum allowed number of actions.
        step = int(self.max_moves - self.agent.moves_remaining)
        time = step / self.max_moves

        return MathyObservation(
            nodes=vectors,
            mask=move_mask,
            values=values,
            type=self.problem_hash(),
            time=[time],
            rnn_state=rnn_state,
            rnn_history=rnn_history,
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


class MathyAgentState:
    """The state related to an agent for a given environment state"""

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
        return MathyAgentState(
            moves_remaining=from_state.moves_remaining,
            problem=from_state.problem,
            reward=from_state.reward,
            problem_type=from_state.problem_type,
            history=from_state.history,
            focus_index=from_state.focus_index,
            last_action=from_state.last_action,
        )
