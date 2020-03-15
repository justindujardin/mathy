from enum import IntEnum
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
import srsly

from .core.expressions import ConstantExpression, MathExpression, MathTypeKeys
from .core.parser import ExpressionParser
from .util import pad_array

PROBLEM_TYPE_HASH_BUCKETS = 128

NodeIntList = List[int]
NodeValuesFloatList = List[float]
NodeMaskIntList = List[int]
ProblemTypeIntList = List[int]


WindowNodeIntList = List[NodeIntList]
WindowNodeMaskIntList = List[NodeMaskIntList]
WindowNodeValuesFloatList = List[NodeValuesFloatList]
WindowProblemTypeIntList = List[ProblemTypeIntList]


# Input type for mathy models
MathyInputsType = List[Any]


class ObservationFeatureIndices(IntEnum):
    nodes = 0
    mask = 1
    values = 2
    type = 3


class MathyObservation(NamedTuple):
    """A featurized observation from an environment state."""

    nodes: NodeIntList
    mask: NodeMaskIntList
    values: NodeValuesFloatList
    type: ProblemTypeIntList


# fmt: off
MathyObservation.nodes.__doc__ = "tree node types in the current environment state shape=[n,]" # noqa
MathyObservation.mask.__doc__ = "0/1 mask where 0 indicates an invalid action shape=[n,]" # noqa
MathyObservation.values.__doc__ = "tree node value sequences, with non number indices set to 0.0 shape=[n,]" # noqa
MathyObservation.type.__doc__ = "two column hash of problem environment type shape=[2,]" # noqa
# fmt: on


class MathyWindowObservation(NamedTuple):
    """A featurized observation from an n-step sequence of environment states."""

    nodes: WindowNodeIntList
    mask: WindowNodeMaskIntList
    values: WindowNodeValuesFloatList
    type: WindowProblemTypeIntList

    def to_inputs(self, as_tf_tensor: bool = False) -> MathyInputsType:
        if as_tf_tensor:
            import tensorflow as tf

        def to_res(in_value):
            if as_tf_tensor is True:
                return tf.convert_to_tensor(in_value, dtype=tf.float32)
            return np.asarray(in_value, dtype="float32")

        result = [
            to_res(self.nodes),
            to_res(self.mask),
            to_res(self.values),
            to_res(self.type),
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
        ]
        return result


# fmt: off
MathyWindowObservation.nodes.__doc__ = "n-step list of node sequences `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.mask.__doc__ = "n-step list of node sequence masks `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.values.__doc__ = "n-step list of value sequences, with non number indices set to 0.0 `shape=[n, max(len(s))]`" # noqa
MathyWindowObservation.type.__doc__ = "n-step problem type hash `shape=[n, 2]`" # noqa
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


_problem_hash_cache: Optional[Dict[str, List[int]]] = None


def observations_to_window(
    observations: List[MathyObservation], total_length: int = None
) -> MathyWindowObservation:
    """Combine a sequence of observations into an observation window"""
    output = MathyWindowObservation(nodes=[], mask=[], values=[], type=[])
    sequence_lengths: List[int] = []
    max_mask_length: int = 0
    for i in range(len(observations)):
        obs = observations[i]
        sequence_lengths.append(len(obs.nodes))
        sequence_lengths.append(len(obs.values))
        max_mask_length = max(max_mask_length, len(obs.mask))
    max_length: int = max(sequence_lengths)
    if total_length is not None:
        max_length = total_length

    for obs in observations:
        output.nodes.append(pad_array(obs.nodes, max_length, MathTypeKeys["empty"]))
        output.mask.append(pad_array(obs.mask, max_mask_length, 0))
        output.values.append(pad_array(obs.values, max_length, 0.0))
        output.type.append(pad_array([], max_length, obs.type))
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
        self.max_moves = max_moves
        self.num_rules = num_rules
        if problem is not None:
            self.agent = MathyAgentState(max_moves, problem, problem_type)
        elif state is not None:
            self.num_rules = state.num_rules
            self.max_moves = state.max_moves
            self.agent = MathyAgentState.copy(state.agent)

    @classmethod
    def copy(cls, from_state):
        return MathyEnvState(state=from_state)

    def clone(self):
        return MathyEnvState(state=self)

    def get_out_state(
        self, problem: str, focus: int, action: int, moves_remaining: int
    ) -> "MathyEnvState":
        """Get the next environment state based on the current one with updated
        history and agent information based on an action being taken."""
        out_state = MathyEnvState.copy(self)
        agent = out_state.agent
        agent.history.append(MathyEnvStateStep(problem, focus, action))
        agent.problem = problem
        agent.action = action
        agent.moves_remaining = moves_remaining
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

    def to_start_observation(self) -> MathyObservation:
        """Generate an episode start MathyObservation"""
        num_actions = 1 * self.num_rules
        hash = self.get_problem_hash()
        mask = [0] * num_actions
        values = [0.0]
        return MathyObservation(
            nodes=[MathTypeKeys["empty"]], mask=mask, values=values, type=hash,
        )

    def to_observation(
        self,
        move_mask: Optional[NodeMaskIntList] = None,
        hash_type: Optional[ProblemTypeIntList] = None,
        parser: Optional[ExpressionParser] = None,
    ) -> MathyObservation:
        """Convert a state into an observation"""
        if parser is None:
            parser = ExpressionParser()
        if hash_type is None:
            hash_type = self.get_problem_hash()
        expression = parser.parse(self.agent.problem)
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

        return MathyObservation(
            nodes=vectors, mask=move_mask, values=values, type=hash_type
        )

    @classmethod
    def from_string(cls, input_string: str) -> "MathyEnvState":
        """Convert a string representation of state into a state object"""
        sep = "@"
        history_sep = ","
        inputs = input_string.split(sep)
        state = MathyEnvState()
        state.max_moves = int(inputs[0])
        state.num_rules = int(inputs[1])
        state.agent = MathyAgentState(
            moves_remaining=int(inputs[2]),
            problem=str(inputs[3]),
            problem_type=str(inputs[4]),
            reward=float(inputs[5]),
            history=[],
        )
        history = inputs[6:]
        for step in history:
            raw, focus, action = step.split(history_sep)
            state.agent.history.append(MathyEnvStateStep(raw, int(focus), int(action)))
        return state

    def to_string(self) -> str:
        """Convert a state object into a string representation"""
        sep = "@"
        out = [
            str(self.max_moves),
            str(self.num_rules),
            str(self.agent.moves_remaining),
            str(self.agent.problem),
            str(self.agent.problem_type),
            str(self.agent.reward),
        ]
        for step in self.agent.history:
            out.append(",".join([str(step.raw), str(step.focus), str(step.action)]))
        return sep.join(out)

    @classmethod
    def from_np(cls, input_bytes: np.ndarray) -> "MathyEnvState":
        """Convert a numpy object into a state object"""
        input_string = "".join([chr(o) for o in input_bytes.tolist()])
        state = cls.from_string(input_string)
        return state

    def to_np(self) -> np.ndarray:
        """Convert a state object into a numpy representation"""
        return np.array([ord(c) for c in self.to_string()])


class MathyAgentState:
    """The state related to an agent for a given environment state"""

    moves_remaining: int
    problem: str
    problem_type: str
    reward: float
    history: List[MathyEnvStateStep]

    def __init__(
        self, moves_remaining, problem, problem_type, reward=0.0, history=None,
    ):
        self.moves_remaining = moves_remaining
        self.problem = problem
        self.reward = reward
        self.problem_type = problem_type
        self.history = (
            history[:] if history is not None else [MathyEnvStateStep(problem, -1, -1)]
        )

    @classmethod
    def copy(cls, from_state: "MathyAgentState"):
        return MathyAgentState(
            moves_remaining=from_state.moves_remaining,
            problem=from_state.problem,
            reward=from_state.reward,
            problem_type=from_state.problem_type,
            history=from_state.history,
        )
