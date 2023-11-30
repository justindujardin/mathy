import copy
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple, Union

import numpy

from .utils import (
    float_type,
    hash_numpy,
    hash_type,
    Scalar,
    similiar_chunks_indexes,
    StateDict,
)


class States:
    """
    Handles several arrays that will contain the data associated with the \
    walkers of a :class:`Swarm`. Each array will be associated to a class \
    attribute, and it will store the corresponding value of that attribute \
    for all the walkers of a :class:`Swarm`.

    This class behaves as a dictionary of arrays with some extra functionality \
    to make easier the process of cloning the walkers' data. All of its internal \
    arrays will have an extra first dimension equal to the number of walkers.

    In order to define the tensors, a `state_dict` dictionary needs to be \
    specified using the following structure::

        state_dict = {"name_1": {"size": tuple([1]),
                                 "dtype": numpy.float32,
                                },
                     }

    Where tuple is a tuple indicating the shape of the desired tensor. The \
    created arrays will accessible the ``name_1`` attribute of the class, or \
    indexing the class with ``states["name_1"]``.

    If ``size`` is not defined the attribute will be considered a vector of \
    length `batch_size`.


    Args:
        batch_size: The number of items in the first dimension of the tensors.
        state_dict: Dictionary defining the attributes of the tensors.
        **kwargs: Data can be directly specified as keyword arguments.

    """

    def __init__(
        self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs
    ):
        """
        Initialize a :class:`States`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        attr_dict = (
            self.params_to_arrays(state_dict, batch_size)
            if state_dict is not None
            else {}
        )
        attr_dict.update(kwargs)
        self._names = list(attr_dict.keys())
        self._attr_dict = attr_dict
        self.update(**self._attr_dict)
        self._batch_size = batch_size

    def __len__(self):
        """Length is equal to n_walkers."""
        return self._batch_size

    def __getitem__(
        self, item: Union[str, int, numpy.int64]
    ) -> Union[numpy.ndarray, List[numpy.ndarray], "States"]:
        """
        Query an attribute of the class as if it was a dictionary.

        Args:
            item: Name of the attribute to be selected.

        Returns:
            The corresponding item.

        """
        if isinstance(item, str):
            try:
                return getattr(self, item)
            except AttributeError:
                raise TypeError(
                    "Tried to get a non existing attribute with key {}".format(item)
                )
        elif isinstance(item, (int, numpy.int64)):
            return self._ix(item)
        else:
            raise TypeError(
                "item must be an instance of str, got {} of type {} instead".format(
                    item, type(item)
                )
            )

    def _ix(self, index: int):
        # TODO(guillemdb): Allow slicing
        data = {
            k: numpy.array([v[index]]) if isinstance(v, numpy.ndarray) else v
            for k, v in self.items()
        }
        return self.__class__(batch_size=1, **data)

    def __setitem__(self, key, value: Union[Tuple, List, numpy.ndarray]):
        """
        Allow the class to set its attributes as if it was a dict.

        Args:
            key: Attribute to be set.
            value: Value of the target attribute.

        Returns:
            None.

        """
        if key not in self._names:
            self._names.append(key)
        self.update(**{key: value})

    def __repr__(self):
        string = "{} with {} walkers\n".format(self.__class__.__name__, self.n)
        for k, v in self.items():
            shape = v.shape if hasattr(v, "shape") else None
            new_str = "{}: {} {}\n".format(k, type(v), shape)
            string += new_str
        return string

    def __hash__(self) -> int:
        _hash = hash(
            tuple(
                [
                    hash_numpy(x) if isinstance(x, numpy.ndarray) else hash(x)
                    for x in self.vals()
                ]
            )
        )
        return _hash

    def group_hash(self, name: str) -> int:
        """Return a unique id for a given attribute."""
        val = getattr(self, name)
        return hash_numpy(val) if isinstance(val, numpy.ndarray) else hash(val)

    def hash_values(self, name: str) -> List[int]:
        """Return a unique id for each walker attribute."""
        values = getattr(self, name)
        hashes = [
            hash_numpy(val) if isinstance(val, numpy.ndarray) else hash(val)
            for val in values
        ]
        return hashes

    @staticmethod
    def merge_states(states: Iterable["States"]) -> "States":
        """
        Combine different states containing the same kind of data into a single \
        :class:`State` with batch size equal to the sum of all the state batch \
        sizes.

        Args:
            states: Iterable returning :class:`States` with the same attributes.

        Returns:
            :class:`States` containing the combined data of the input values.

        """

        def merge_one_name(states_list, name):
            vals = []
            is_scalar_vector = True
            for state in states_list:
                data = state[name]
                # Attributes that are not numpy arrays are not stacked.
                if not isinstance(data, numpy.ndarray):
                    return data
                state_len = len(state)
                if len(data.shape) == 0 and state_len == 1:
                    # Name is scalar vector. Data is scalar value. Transform to array first
                    value = numpy.array([data]).flatten()
                elif len(data.shape) == 1 and state_len == 1:
                    if data.shape[0] == 1:
                        # Name is scalar vector. Data already transformed to an array
                        value = data
                    else:
                        # Name is a matrix of vectors. Data needs an additional dimension
                        is_scalar_vector = False
                        value = numpy.array([data])
                elif len(data.shape) == 1 and state_len > 1:
                    # Name is a scalar vector. Data already has is a one dimensional array
                    value = data
                elif (
                    len(data.shape) > 1
                    and state_len > 1
                    or len(data.shape) > 1
                    and len(state) == 1
                ):
                    # Name is a matrix of vectors. Data has the correct shape
                    is_scalar_vector = False
                    value = data
                else:
                    raise ValueError(
                        "Could not infer data concatenation for attribute %s  with shape %s"
                        % (name, data.shape)
                    )
                vals.append(value)
            if is_scalar_vector:
                return numpy.concatenate(vals)
            else:
                return numpy.vstack(vals)

        # Assumes all states have the same names.
        data = {name: merge_one_name(states, name) for name in states[0]._names}
        batch_size = sum(s.n for s in states)
        return states[0].__class__(batch_size=batch_size, **data)

    @property
    def n(self) -> int:
        """Return the batch_size of the vectors, which is equivalent to the number of walkers."""
        return self._batch_size

    def get(self, key: str, default=None):
        """
        Get an attribute by key and return the default value if it does not exist.

        Args:
            key: Attribute to be recovered.
            default: Value returned in case the attribute is not part of state.

        Returns:
            Target attribute if found in the instance, otherwise returns the
             default value.

        """
        if key not in self.keys():
            return default
        return self[key]

    def keys(self) -> Generator:
        """Return a generator for the attribute names of the stored data."""
        return (name for name in self._names if not name.startswith("_"))

    def vals(self) -> Generator:
        """Return a generator for the values of the stored data."""
        return (self[name] for name in self._names if not name.startswith("_"))

    def items(self) -> Generator:
        """Return a generator for the attribute names and the values of the stored data."""
        return ((name, self[name]) for name in self._names if not name.startswith("_"))

    def itervals(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if self.n <= 1:
            return self.vals()
        for i in range(self.n):
            yield tuple([v[i] for v in self.vals()])

    def iteritems(self):
        """
        Iterate the states attributes by walker.

        Returns:
            Tuple containing all the names of the attributes, and the values that
            correspond to a given walker.

        """
        if self.n < 1:
            return self.vals()
        for i in range(self.n):
            values = (v[i] if isinstance(v, numpy.ndarray) else v for v in self.vals())
            yield tuple(self._names), tuple(values)

    def split_states(self, n_chunks: int) -> Generator["States", None, None]:
        """
        Return a generator for n_chunks different states, where each one \
        contain only the data corresponding to one walker.
        """

        def get_chunck_size(state, start, end):
            for name in state._names:
                attr = state[name]
                if isinstance(attr, numpy.ndarray):
                    return len(attr[start:end])
            return int(numpy.ceil(self.n / n_chunks))

        for start, end in similiar_chunks_indexes(self.n, n_chunks):
            chunk_size = get_chunck_size(self, start, end)

            data = {
                k: val[start:end] if isinstance(val, numpy.ndarray) else val
                for k, val in self.items()
            }
            new_state = self.__class__(batch_size=chunk_size, **data)
            yield new_state

    def update(self, other: "States" = None, **kwargs):
        """
        Modify the data stored in the States instance.

        Existing attributes will be updated, and new attributes will be created if needed.

        Args:
            other: State class that will be copied upon update.
            **kwargs: It is possible to specify the update as key value attributes, \
                     where key is the name of the attribute to be updated, and value \
                      is the new value for the attribute.
        """

        def update_or_set_attributes(attrs: Union[dict, States]):
            for name, val in attrs.items():
                try:
                    getattr(self, name)[:] = copy.deepcopy(val)
                except (AttributeError, TypeError, KeyError, ValueError):
                    setattr(self, name, copy.deepcopy(val))

        if other is not None:
            update_or_set_attributes(other)
        if kwargs:
            update_or_set_attributes(kwargs)

    def clone(
        self,
        will_clone: numpy.ndarray,
        compas_ix: numpy.ndarray,
        ignore: Optional[Set[str]] = None,
    ):
        """
        Clone all the stored data according to the provided arrays.

        Args:
            will_clone: Array of shape (n_walkers,) of booleans indicating the \
                        index of the walkers that will clone to a random companion.
            compas_ix: Array of integers of shape (n_walkers,). Contains the \
                       indexes of the walkers that will be copied.
            ignore: set containing the names of the attributes that will not be \
                    cloned.

        """
        ignore = set() if ignore is None else ignore
        for name in self.keys():
            if isinstance(self[name], numpy.ndarray) and name not in ignore:
                self[name][will_clone] = self[name][compas_ix][will_clone]

    def get_params_dict(self) -> StateDict:
        """Return a dictionary describing the data stored in the :class:`States`."""
        return {
            k: {"shape": v.shape, "dtype": v.dtype}
            for k, v in self.__dict__.items()
            if isinstance(v, numpy.ndarray)
        }

    def copy(self) -> "States":
        """Crete a copy of the current instance."""
        param_dict = {str(name): val.copy() for name, val in self.items()}
        return States(batch_size=self.n, **param_dict)

    @staticmethod
    def params_to_arrays(
        param_dict: StateDict, n_walkers: int
    ) -> Dict[str, numpy.ndarray]:
        """
        Create a dictionary containing the arrays specified by param_dict.

        Args:
            param_dict: Dictionary defining the attributes of the tensors.
            n_walkers: Number items in the first dimension of the data tensors.

        Returns:
              Dictionary with the same keys as param_dict, containing arrays specified \
              by `param_dict` values.

        """
        tensor_dict = {}
        for key, val in param_dict.items():
            # Shape already includes the number of walkers. Remove walkers axis to create size.
            shape = val.get("shape")
            if shape is None:
                val_size = val.get("size")
            elif len(shape) > 1:
                val_size = shape[1:]
            else:
                val_size = val.get("size")
            # Create appropriate shapes with current state's number of walkers.
            sizes = n_walkers if val_size is None else tuple([n_walkers]) + val_size
            if "size" in val:
                del val["size"]
            if "shape" in val:
                del val["shape"]
            tensor_dict[key] = numpy.zeros(shape=sizes, **val)
        return tensor_dict


class StatesEnv(States):
    """
    Keeps track of the data structures used by the :class:`Environment`.

    Attributes:
        states: This data tracks the internal state of the Environment simulation, \
                 and they are only used to save and restore its state.
        observs: This is the data that corresponds to the observations of the \
                 current :class:`Environment` state. The observations are used \
                 for calculating distances.
        rewards: This vector contains the rewards associated with each observation.
        oobs: Stands for **Out Of Bounds**. It is a vector of booleans that \
              represents and arbitrary boundary condition. If a value is ``True`` \
              the corresponding states will be treated as being outside the \
              :class:`Environment` domain. The states considered out of bounds \
              will be avoided by the sampling algorithms.
        terminals: Vector of booleans representing the successful termination \
                   of an environment. A ``True`` value indicates that the \
                   :class:`Environment` has successfully reached a terminal \
                   state that is not out of bounds.

    """

    def __init__(
        self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs
    ):
        """
        Initialise a :class:`StatesEnv`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        self.observs = None
        self.states = None
        self.rewards = None
        self.oobs = None
        self.terminals = None
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesEnv, self).__init__(
            state_dict=updated_dict, batch_size=batch_size, **kwargs
        )

    def get_params_dict(self) -> StateDict:
        """Return a dictionary describing the data stored in the :class:`StatesEnv`."""
        params = {
            "states": {"dtype": numpy.int64},
            "observs": {"dtype": numpy.float32},
            "rewards": {"dtype": numpy.float32},
            "oobs": {"dtype": numpy.bool_},
            "terminals": {"dtype": numpy.bool_},
        }
        state_dict = super(StatesEnv, self).get_params_dict()
        params.update(state_dict)
        return params


class StatesModel(States):
    """
    Keeps track of the data structures used by the :class:`Model`.

    Attributes:
        actions: Represents the actions that will be sampled by a :class:`Model`.

    """

    def __init__(
        self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs
    ):
        """
        Initialise a :class:`StatesModel`.

        Args:
             batch_size: The number of items in the first dimension of the tensors.
             state_dict: Dictionary defining the attributes of the tensors.
             **kwargs: The name-tensor pairs can also be specified as kwargs.

        """
        self.actions = None
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesModel, self).__init__(
            state_dict=updated_dict, batch_size=batch_size, **kwargs
        )

    def get_params_dict(self) -> StateDict:
        """Return the parameter dictionary with tre attributes common to all Models."""
        params = {
            "actions": {"dtype": numpy.float32},
        }
        state_dict = super(StatesModel, self).get_params_dict()
        params.update(state_dict)
        return params


class StatesWalkers(States):
    """
    Keeps track of the data structures used by the :class:`Walkers`.

    Attributes:
        id_walkers: Array of of integers that uniquely identify a given state. \
                    They are obtained by hashing the states.
        compas_clone: Array of integers containing the index of the walkers \
                      selected as companions for cloning.
        processed_rewards: Array of normalized rewards. It contains positive \
                           values with an average of 1. Values greater than one \
                           correspond to rewards above the average, and values \
                           lower than one correspond to rewards below the average.
        virtual_rewards: Array containing the virtual rewards assigned to each walker.
        cum_rewards: Array of rewards used to compute the virtual_reward. This \
                    value can accumulate the rewards provided by the \
                    :class:`Environment` during an algorithm run.
        distances: Array containing the similarity metric of each walker used \
                   to compute the virtual reward.
        clone_probs: Array containing the probability that a walker clones to \
                     its companion during the cloning phase.
        will_clone: Boolean array. A ``True`` value indicates that the \
                    corresponding walker will clone to its companion.
        in_bounds: Boolean array. A `True` value indicates that a walker is \
                   in the domain defined by the :class:`Environment`.

        best_state: State of the walker with the best ``cum_reward`` found \
                   during the algorithm run.
        best_obs: Observation corresponding to the ``best_state``.
        best_reward: Best ``cum_reward`` found during the algorithm run.
        best_id: Integer representing the hash of the ``best_state``.

    """

    def __init__(
        self, batch_size: int, state_dict: Optional[StateDict] = None, **kwargs
    ):
        """
        Initialize a :class:`StatesWalkers`.

        Args:
            batch_size: Number of walkers that the class will be tracking.
            state_dict: Dictionary defining the attributes of the tensors.
            kwargs: attributes that will not be set as numpy.ndarrays
        """
        self.will_clone = None
        self.compas_clone = None
        self.processed_rewards = None
        self.cum_rewards = None
        self.virtual_rewards = None
        self.distances = None
        self.clone_probs = None
        self.in_bounds = None
        self.id_walkers = None
        # This is only to allow __repr__. Should be overridden after reset
        self.best_id = None
        self.best_obs = None
        self.best_state = None
        self.best_reward = -numpy.inf
        updated_dict = self.get_params_dict()
        if state_dict is not None:
            updated_dict.update(state_dict)
        super(StatesWalkers, self).__init__(
            state_dict=updated_dict, batch_size=batch_size, **kwargs
        )

    def get_params_dict(self) -> StateDict:
        """Return a dictionary containing the param_dict to build an instance \
        of States that can handle all the data generated by the :class:`Walkers`.
        """
        params = {
            "id_walkers": {"dtype": hash_type},
            "compas_clone": {"dtype": numpy.int64},
            "processed_rewards": {"dtype": float_type},
            "virtual_rewards": {"dtype": float_type},
            "cum_rewards": {"dtype": float_type},
            "distances": {"dtype": float_type},
            "clone_probs": {"dtype": float_type},
            "will_clone": {"dtype": numpy.bool_},
            "in_bounds": {"dtype": numpy.bool_},
        }
        state_dict = super(StatesWalkers, self).get_params_dict()
        params.update(state_dict)
        return params

    def clone(self, **kwargs) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Perform the clone only on cum_rewards and id_walkers and reset the other arrays."""
        clone, compas = self.will_clone, self.compas_clone
        self.cum_rewards[clone] = copy.deepcopy(self.cum_rewards[compas][clone])
        self.id_walkers[clone] = copy.deepcopy(self.id_walkers[compas][clone])
        return clone, compas

    def reset(self):
        """Clear the internal data of the class."""
        params = self.get_params_dict()
        other_attrs = [name for name in self.keys() if name not in params]
        for attr in other_attrs:
            setattr(self, attr, None)
        self.update(
            id_walkers=numpy.zeros(self.n, dtype=hash_type),
            compas_dist=numpy.arange(self.n),
            compas_clone=numpy.arange(self.n),
            processed_rewards=numpy.zeros(self.n, dtype=float_type),
            cum_rewards=numpy.zeros(self.n, dtype=float_type),
            virtual_rewards=numpy.ones(self.n, dtype=float_type),
            distances=numpy.zeros(self.n, dtype=float_type),
            clone_probs=numpy.zeros(self.n, dtype=float_type),
            will_clone=numpy.zeros(self.n, dtype=numpy.bool_),
            in_bounds=numpy.ones(self.n, dtype=numpy.bool_),
        )

    def _ix(self, index: int):
        # TODO(guillemdb): Allow slicing
        data = {
            k: numpy.array([v[index]])
            if isinstance(v, numpy.ndarray) and "best" not in k
            else v
            for k, v in self.items()
        }
        return self.__class__(batch_size=1, **data)


class OneWalker(States):
    """
    Represent one walker.

    This class is used for initializing a :class:`Swarm` to a given state without having to
    explicitly define the :class:`StatesEnv`, :class:`StatesModel` and :class:`StatesWalkers`.

    """

    def __init__(
        self,
        state: numpy.ndarray,
        observ: numpy.ndarray,
        reward: Scalar,
        id_walker=None,
        state_dict: StateDict = None,
        **kwargs
    ):
        """
        Initialize a :class:`OneWalker`.

        Args:
            state: Non batched numpy array defining the state of the walker.
            observ: Non batched numpy array defining the observation of the walker.
            reward: Scalar value representing the reward of the walker.
            id_walker: Hash of the provided State. If None it will be calculated when the
                       the :class:`OneWalker` is initialized.
            state_dict: External :class:`StateDict` that overrides the default values.
            **kwargs: Additional data needed to define the walker. Its structure \
                      needs to be defined in the provided ``state_dict``. These attributes
                      will be assigned to the :class:`EnvStates` of the :class:`Swarm`.

        """
        self.id_walkers = None
        self.rewards = None
        self.observs = None
        self.states = None
        self._observs_size = observ.shape
        self._observs_dtype = observ.dtype
        self._states_size = state.shape
        self._states_dtype = state.dtype
        self._rewards_dtype = type(reward)
        # Accept external definition of param_dict values
        walkers_dict = self.get_params_dict()
        if state_dict is not None:
            for k, v in state_dict.items():
                if k in [
                    "observs",
                    "states",
                ]:  # These two are parsed from the provided opts
                    continue
                if k in walkers_dict:
                    walkers_dict[k] = v
        super(OneWalker, self).__init__(batch_size=1, state_dict=walkers_dict)
        # Keyword arguments must be defined in state_dict
        if state_dict is not None:
            for k in kwargs.keys():
                if k not in state_dict:
                    raise ValueError(
                        "The provided attributes must be defined in state_dict."
                        "param_dict: %s\n kwargs: %s" % (state_dict, kwargs)
                    )
        self.observs[:] = copy.deepcopy(observ)
        self.states[:] = copy.deepcopy(state)
        self.rewards[:] = copy.deepcopy(reward)
        self.id_walkers[:] = (
            copy.deepcopy(id_walker) if id_walker is not None else hash_numpy(state)
        )
        self.update(**kwargs)

    def __repr__(self):
        with numpy.printoptions(linewidth=100, threshold=200, edgeitems=9):
            string = (
                "reward: %s\n"
                "observ: %s\n"
                "state: %s\n"
                "id: %s"
                % (
                    self.rewards[0],
                    self.observs[0].flatten(),
                    self.states[0].flatten(),
                    self.id_walkers[0],
                )
            )
            return string

    def get_params_dict(self) -> StateDict:
        """Return a dictionary containing the param_dict to build an instance \
        of States that can handle all the data generated by the :class:`Walkers`.
        """
        params = {
            "id_walkers": {"dtype": hash_type},
            "rewards": {"dtype": self._rewards_dtype},
            "observs": {"dtype": self._observs_dtype, "size": self._observs_size},
            "states": {"dtype": self._states_dtype, "size": self._states_size},
        }
        return params
