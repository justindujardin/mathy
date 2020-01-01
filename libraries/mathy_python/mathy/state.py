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
RNNStateFloatList = List[float]


WindowNodeIntList = List[NodeIntList]
WindowNodeMaskIntList = List[NodeMaskIntList]
WindowNodeValuesFloatList = List[NodeValuesFloatList]
WindowProblemTypeIntList = List[ProblemTypeIntList]
WindowProblemTimeFloatList = List[ProblemTimeFloatList]
WindowRNNStateFloatList = List[RNNStateFloatList]


# Input type for mathy models
MathyInputsType = List[Any]


class ObservationFeatureIndices(IntEnum):
    nodes = 0
    mask = 1
    values = 2
    type = 3
    time = 4
    rnn_state_h = 5
    rnn_state_c = 6
    rnn_history_h = 7


class MathyObservation(NamedTuple):
    """A featurized observation from an environment state."""

    nodes: NodeIntList
    mask: NodeMaskIntList
    values: NodeValuesFloatList
    type: ProblemTypeIntList
    time: ProblemTimeFloatList
    rnn_state_h: RNNStateFloatList
    rnn_state_c: RNNStateFloatList
    rnn_history_h: RNNStateFloatList


# fmt: off
MathyObservation.nodes.__doc__ = "tree node types in the current environment state shape=[n,]" # noqa
MathyObservation.mask.__doc__ = "0/1 mask where 0 indicates an invalid action shape=[n,]" # noqa
MathyObservation.values.__doc__ = "tree node value sequences, with non number indices set to 0.0 shape=[n,]" # noqa
MathyObservation.type.__doc__ = "two column hash of problem environment type shape=[2,]" # noqa
MathyObservation.time.__doc__ = "float value between 0.0-1.0 for how much time has passed shape=[1,]" # noqa
MathyObservation.rnn_state_h.__doc__ = "rnn state_h shape=[dimensions]" # noqa
MathyObservation.rnn_state_c.__doc__ = "rnn state_c shape=[dimensions]" # noqa
MathyObservation.rnn_history_h.__doc__ = "rnn historical state_h shape=[dimensions]" # noqa
# fmt: on


class MathyWindowObservation(NamedTuple):
    """A featurized observation from an n-step sequence of environment states."""

    nodes: WindowNodeIntList
    mask: WindowNodeMaskIntList
    values: WindowNodeValuesFloatList
    type: WindowProblemTypeIntList
    time: WindowProblemTimeFloatList
    rnn_state_h: WindowRNNStateFloatList
    rnn_state_c: WindowRNNStateFloatList
    rnn_history_h: WindowRNNStateFloatList

    def to_inputs(self) -> MathyInputsType:
        import tensorflow as tf

        result = [
            tf.convert_to_tensor(self.nodes),
            tf.convert_to_tensor(self.mask),
            tf.convert_to_tensor(self.values),
            tf.convert_to_tensor(self.type),
            tf.convert_to_tensor(self.time),
            tf.convert_to_tensor(self.rnn_state_h),
            tf.convert_to_tensor(self.rnn_state_c),
            tf.convert_to_tensor(self.rnn_history_h),
        ]
        for r in result:
            for s in r.shape:
                assert s is not None
        return result

    def to_input_shapes(self) -> List[Any]:
        import tensorflow as tf

        result = [
            np.asarray(self.nodes).shape,
            np.asarray(self.mask).shape,
            np.asarray(self.values).shape,
            np.asarray(self.type).shape,
            np.asarray(self.time).shape,
            np.asarray(self.rnn_state_h).shape,
            np.asarray(self.rnn_state_c).shape,
            np.asarray(self.rnn_history_h).shape,
        ]
        return result


# fmt: off
MathyWindowObservation.nodes.__doc__ = "n-step list of node sequences `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.mask.__doc__ = "n-step list of node sequence masks `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.values.__doc__ = "n-step list of value sequences, with non number indices set to 0.0 `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.type.__doc__ = "n-step problem type hash `shape=[n, 2]`" # noqa
MathyWindowObservation.time.__doc__ = "float value between 0.0-1.0 for how much time has passed `shape=[n, 1]`" # noqa
MathyWindowObservation.rnn_state_h.__doc__ = "n-step rnn state `shape=[n, dimensions]`" # noqa
MathyWindowObservation.rnn_state_c.__doc__ = "n-step rnn state `shape=[n, dimensions]`" # noqa
MathyWindowObservation.rnn_history_h.__doc__ = "n-step rnn historical state `shape=[n, dimensions]`" # noqa
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


_cached_placeholder: Optional[RNNStateFloatList] = None
_problem_hash_cache: Optional[Dict[str, List[int]]] = None


def rnn_placeholder_state(rnn_size: int) -> RNNStateFloatList:
    """Create a placeholder state for the RNN hidden states in an observation. This
    is useful because we don't always know the RNN state when we initialize an
    observation. """
    global _cached_placeholder
    # Not in cache, or rnn_size differs
    if _cached_placeholder is None or len(_cached_placeholder) != rnn_size:
        _cached_placeholder = [0.0] * rnn_size
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
        rnn_state_h=[],
        rnn_state_c=[],
        rnn_history_h=[],
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
        output.type.append(pad_array([], max_length, obs.type))
        output.time.append(pad_array([], max_length, obs.time))
        output.rnn_state_h.append(obs.rnn_state_h)
        output.rnn_state_c.append(obs.rnn_state_c)
        output.rnn_history_h.append(obs.rnn_history_h)
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
                int(problem_hash_one.numpy()),
                int(problem_hash_two.numpy()),
            ]
        return _problem_hash_cache[self.agent.problem_type]

    def to_start_observation(
        self, rnn_state_h: RNNStateFloatList, rnn_state_c: RNNStateFloatList
    ) -> MathyObservation:
        """Generate an episode start MathyObservation"""
        num_actions = 1 * self.num_rules
        hash = self.get_problem_hash()
        mask = [0] * num_actions
        # HACKS: if you swap this line with below and train an agent on "poly,complex,binomial"
        #        it will crash after a few episodes, allowing you to test the error printing with
        #        call stack print outs.
        # values = [0.0] * num_actions
        values = [0.0]
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]],
            mask=mask,
            values=values,
            type=hash,
            time=[-1.0],
            rnn_state_h=rnn_state_h,
            rnn_state_c=rnn_state_c,
            rnn_history_h=rnn_state_h,
        )

    def to_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        rnn_state_h: Optional[RNNStateFloatList] = None,
        rnn_state_c: Optional[RNNStateFloatList] = None,
        rnn_history_h: Optional[RNNStateFloatList] = None,
        hash_type: Optional[ProblemTypeIntList] = None,
    ) -> MathyObservation:
        if rnn_state_h is None:
            rnn_state_h = rnn_placeholder_state(128)
        if rnn_state_c is None:
            rnn_state_c = rnn_placeholder_state(128)
        if rnn_history_h is None:
            rnn_history_h = rnn_placeholder_state(128)
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
            rnn_state_h=rnn_state_h,
            rnn_state_c=rnn_state_c,
            rnn_history_h=rnn_history_h,
        )


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
