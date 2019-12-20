from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional

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


# Input type for mathy models
MathyInputsType = List[Any]


class ObservationFeatureIndices(IntEnum):
    nodes = 0
    mask = 1
    values = 2
    type = 3
    time = 4
    rnn_state = 5
    rnn_history = 6


class MathyObservation(NamedTuple):
    """A featurized observation from an environment state."""

    nodes: NodeIntList
    mask: NodeMaskIntList
    values: NodeValuesFloatList
    type: ProblemTypeIntList
    time: ProblemTimeFloatList
    rnn_state: RNNStatesFloatList
    rnn_history: RNNStatesFloatList


# fmt: off
MathyObservation.nodes.__doc__ = "tree node types in the current environment state shape=[n,]" # noqa
MathyObservation.mask.__doc__ = "0/1 mask where 0 indicates an invalid action shape=[n,]" # noqa
MathyObservation.values.__doc__ = "tree node value sequences, with non number indices set to 0.0 shape=[n,]" # noqa
MathyObservation.type.__doc__ = "two column hash of problem environment type shape=[2,]" # noqa
MathyObservation.time.__doc__ = "float value between 0.0-1.0 for how much time has passed shape=[1,]" # noqa
MathyObservation.rnn_state.__doc__ = "rnn state pairs shape=[2*dimensions]" # noqa
MathyObservation.rnn_history.__doc__ = "rnn historical state pairs shape=[2*dimensions]" # noqa
# fmt: on


class MathyWindowObservation(NamedTuple):
    """A featurized observation from an n-step sequence of environment states."""

    nodes: WindowNodeIntList
    mask: WindowNodeMaskIntList
    values: WindowNodeValuesFloatList
    type: WindowProblemTypeIntList
    time: WindowProblemTimeFloatList
    rnn_state: WindowRNNStatesFloatList
    rnn_history: WindowRNNStatesFloatList

    def to_inputs(self) -> MathyInputsType:
        import tensorflow as tf

        result = [
            tf.convert_to_tensor(self.nodes),
            tf.convert_to_tensor(self.mask),
            tf.convert_to_tensor(self.values),
            tf.convert_to_tensor(self.type),
            tf.convert_to_tensor(self.time),
            tf.convert_to_tensor([self.rnn_state]),
            tf.convert_to_tensor([self.rnn_history]),
        ]
        for r in result:
            for s in r.shape:
                assert s is not None
        return result

    def to_input_shapes(self) -> List[Any]:
        import tensorflow as tf

        result = [
            tf.convert_to_tensor(self.nodes).shape,
            tf.convert_to_tensor(self.mask).shape,
            tf.convert_to_tensor(self.values).shape,
            tf.convert_to_tensor(self.type).shape,
            tf.convert_to_tensor(self.time).shape,
            tf.convert_to_tensor([self.rnn_state]).shape,
            tf.convert_to_tensor([self.rnn_history]).shape,
        ]
        return result


# fmt: off
MathyWindowObservation.nodes.__doc__ = "n-step list of node sequences `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.mask.__doc__ = "n-step list of node sequence masks `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.values.__doc__ = "n-step list of value sequences, with non number indices set to 0.0 `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.type.__doc__ = "n-step problem type hash `shape=[n, 2]`" # noqa
MathyWindowObservation.time.__doc__ = "float value between 0.0-1.0 for how much time has passed `shape=[n, 1]`" # noqa
MathyWindowObservation.rnn_state.__doc__ = "n-step rnn state pairs `shape=[n, 2*dimensions]`" # noqa
MathyWindowObservation.rnn_history.__doc__ = "n-step rnn historical state pairs `shape=[n, 2*dimensions]`" # noqa
# fmt: on


class MathyEnvStateStep(NamedTuple):
    """Capture summarized environment state for a previous timestep so the
    agent can use context from its history when making new predictions."""

    raw: str
    focus: int
    action: int


# fmt: off
MathyEnvStateStep.raw.__doc__ = "the input text at the timestep" # noqa
MathyEnvStateStep.focus.__doc__ = "the index of the node that is acted on" # noqa
MathyEnvStateStep.action.__doc__ = "the action taken" # noqa
# fmt: on


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
    sequence_lengths: List[int] = []
    max_mask_length: int = 0
    for i in range(len(observations)):
        obs = observations[i]
        sequence_lengths.append(len(obs.nodes))
        sequence_lengths.append(len(obs.values))
        max_mask_length = max(max_mask_length, len(obs.mask))
    max_length: int = max(sequence_lengths)

    for obs in observations:
        output.nodes.append(pad_array(obs.nodes, max_length, MathTypeKeys["empty"]))
        output.mask.append(pad_array(obs.mask, max_mask_length, 0))
        output.values.append(pad_array(obs.values, max_length, 0.0))
        output.type.append(obs.type)
        output.time.append(obs.time)
        output.rnn_state[0].append(obs.rnn_state[0])
        output.rnn_state[1].append(obs.rnn_state[1])
        output.rnn_history[0].append(obs.rnn_history[0])
        output.rnn_history[1].append(obs.rnn_history[1])
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

    def get_out_state(
        self, problem: str, action: int, focus_index: int, moves_remaining: int
    ) -> "MathyEnvState":
        """Get the next environment state based on the current one with updated
        history and agent information based on an action being taken."""
        out_state = MathyEnvState.copy(self)
        agent = out_state.agent
        agent.history.append(MathyEnvStateStep(problem, focus_index, action))
        agent.problem = problem
        agent.action = action
        agent.moves_remaining = moves_remaining
        agent.focus_index = focus_index
        return out_state

    def get_problem_hash(self) -> ProblemTypeIntList:
        """Return a two element array with hashed values for the current environment
        namespace string.

        # Example

        - `mycorp.envs.solve_impossible_problems` -> `[12375561, -2838517]`

        """
        # NOTE: import delayed to enable MP
        import tensorflow as tf

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
            _problem_hash_cache[self.agent.problem_type] = [
                problem_hash_one,
                problem_hash_two,
            ]
        return _problem_hash_cache[self.agent.problem_type]

    def to_start_observation(self, rnn_state: RNNStatesFloatList) -> MathyObservation:
        """Generate an episode start MathyObservation"""
        num_actions = 1 * self.num_rules
        hash = self.get_problem_hash()
        mask = [0] * num_actions
        values = [0.0]
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]],
            mask=mask,
            values=values,
            type=hash,
            time=[-1.0],
            rnn_state=rnn_state,
            rnn_history=rnn_state,
        )

    def to_empty_observation(self, hash=None, rnn_size: int = 128) -> MathyObservation:
        """Generate an empty MathyObservation. This is different than
        to_start_observation in that uses a zero'd placeholder RNN state
        rather than the given start state from the agent."""
        num_actions = 1 * self.num_rules
        if hash is None:
            hash = self.get_problem_hash()
        mask = [0] * num_actions
        rnn_state = rnn_placeholder_state(rnn_size)
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]],
            mask=mask,
            values=[0.0],
            type=hash,
            time=[0.0],
            rnn_state=rnn_state,
            rnn_history=rnn_state,
        )

    def get_node_vectors(self, expression: MathExpression):
        """Get a set of context-sensitive vectors for a given expression"""
        nodes = expression.to_list()
        vectors = []
        nodes_len = len(nodes)
        pad_value = MathTypeKeys["empty"]
        context_pad_value = (pad_value, pad_value, pad_value)
        # Add context before/current/after to nodes (thanks @honnibal for this trick)
        for i, t in enumerate(nodes):
            last = pad_value if i == 0 else nodes[i - 1].type_id
            next = pad_value if i > nodes_len - 2 else nodes[i + 1].type_id
            vectors.append((last, t.type_id, next))

        vectors_len = len(vectors)

        # Do it again which expands the reach of the vectors.
        context_vectors = []
        for i, v in enumerate(vectors):
            last = context_pad_value if i == 0 else vectors[i - 1]
            next = context_pad_value if i > vectors_len - 2 else vectors[i + 1]
            context_vectors.append(last + v + next)

        return context_vectors

    def to_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        rnn_state: Optional[RNNStatesFloatList] = None,
        rnn_history: Optional[RNNStatesFloatList] = None,
        hash_type: Optional[ProblemTypeIntList] = None,
    ) -> MathyObservation:
        if rnn_state is None:
            rnn_state = rnn_placeholder_state(128)
        if rnn_history is None:
            rnn_history = rnn_placeholder_state(128)
        if hash_type is None:
            hash_type = self.get_problem_hash()
        expression = self.parser.parse(self.agent.problem)
        nodes: List[MathExpression] = expression.to_list()
        vectors: NodeIntList = []
        values: NodeValuesFloatList = []
        if move_mask is None:
            move_mask = np.zeros(len(nodes))
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
            type=hash_type,
            time=[time],
            rnn_state=rnn_state,
            rnn_history=rnn_history,
        )

    def to_empty_window(self, samples: int, rnn_size: int) -> MathyWindowObservation:
        """Generate an empty window of observations that can be passed
        through the model"""
        hash = self.get_problem_hash()
        window = observations_to_window(
            [self.to_empty_observation(hash, rnn_size) for _ in range(samples)]
        )
        return window


class MathyAgentState:
    """The state related to an agent for a given environment state"""

    moves_remaining: int
    problem: str
    problem_type: str
    reward: float
    history: List[MathyEnvStateStep]
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
        self.focus_index = focus_index
        self.problem = problem
        self.last_action = last_action
        self.reward = reward
        self.problem_type = problem_type
        self.history = (
            history[:] if history is not None else [MathyEnvStateStep(problem, -1, -1)]
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
