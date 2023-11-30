import numpy

from fragile.core.models import NormalContinuous
from fragile.core.states import StatesEnv, StatesModel, StatesWalkers


class ESModel(NormalContinuous):
    """
    The ESModel implements an evolutionary strategy policy.

    It mutates randomly some of the coordinates of the best solution found by \
    substituting them with a proposal solution. This proposal solution is the \
    difference between two random permutations of the best solution found.

    It applies a gaussian normal perturbation with a probability given by ``mutation``.
    """

    def __init__(
        self,
        mutation: float = 0.5,
        recombination: float = 0.7,
        random_step_prob: float = 0.1,
        *args,
        **kwargs
    ):
        """
        Initialize a :class:`ESModel`.

        Args:
            mutation: Probability of mutating a coordinate of the solution vector.
            recombination: Step size of the update applied to the best solution found.
            random_step_prob: Probability of applying a random normal perturbation.
            *args: Passed to the parent :class:`NormalContinuous`.
            **kwargs: Passed to the parent :class:`NormalContinuous`.

        """
        super(ESModel, self).__init__(*args, **kwargs)
        self.mutation = mutation
        self.recombination = recombination
        self.random_step_prob = random_step_prob

    def sample(
        self,
        batch_size: int,
        model_states: StatesModel = None,
        env_states: StatesEnv = None,
        walkers_states: StatesWalkers = None,
        **kwargs,
    ) -> StatesModel:
        """
        Calculate the corresponding data to interact with the Environment and \
        store it in model states.

        Args:
            batch_size: Number of new points to the sampled.
            model_states: States corresponding to the environment data.
            env_states: States corresponding to the model data.
            walkers_states: States corresponding to the walkers data.
            kwargs: Passed to the :class:`Critic` if any.

        Returns:
            Tuple containing a tensor with the sampled actions and the new model states variable.

        """
        # There is a chance of performing a gaussian perturbation
        if numpy.random.random() < self.random_step_prob:
            return super(ESModel, self).sample(
                batch_size=batch_size, env_states=env_states, model_states=model_states, **kwargs,
            )
        observs = (
            env_states.observs
            if env_states is not None
            else numpy.zeros(((batch_size,) + self.shape))
        )
        has_best = walkers_states is not None and walkers_states.best_state is not None
        best = walkers_states.best_state if has_best else observs
        # Choose 2 random indices
        indexes = numpy.arange(observs.shape[0])
        a_rand = self.random_state.permutation(indexes)
        b_rand = self.random_state.permutation(indexes)
        proposal = best + self.recombination * (observs[a_rand] - observs[b_rand])
        # Randomly mutate each coordinate of the original vector
        assert observs.shape == proposal.shape
        rands = numpy.random.random(observs.shape)
        perturbations = numpy.where(rands < self.mutation, observs, proposal)
        # The Environment will sum the observations to perform the step
        new_states = perturbations - observs
        actions = self.bounds.clip(new_states)
        return self.update_states_with_critic(
            actions=actions,
            batch_size=batch_size,
            model_states=model_states,
            env_states=env_states,
            walkers_states=walkers_states,
            **kwargs
        )
