# Swarm Planning Solver [![Open Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justindujardin/mathy/blob/master/website/docs/examples/swarm_solver.ipynb)

> This notebook is built using `mathy.fragile` module for swarm planning to determine which actions to take. The [research](https://arxiv.org/abs/1803.05049) and [implementation](https://github.com/FragileTech/FractalAI) come from [@Guillemdb](https://github.com/Guillemdb) and [@sergio-hcsoft](https://github.com/sergio-hcsoft). They're both amazing 🙇

Sometimes training a machine learning model is inconvenient and time consuming, especially when you're working on a new problem type or set of rules. 

So what do we do in these cases? We use mathy's built-in swarm planning algorithm, of course!

When you're developing a rule or environment, you'd often like to know how well an average agent can be expected to perform on this task, without fully-training a model each time you change the code. 

Let's look together at how we can use mathy.fragile to implement an agent that selects winning actions without any training, while still showing its work step-by-step. 


```python
!pip install mathy mathy_core mathy_envs
```

## Fractal Monte Carlo

The Fractal Monte Carlo (FMC) algorithm we use comes from `mathy.fragile` module and uses a swarm of walkers to explore your environment and find optimal paths to the solution. We'll use it with `mathy_envs`, to solve math problems step-by-step.

By the time you're done with this notebook, you should have a general understanding of how FMC, through its unique path-search capabilities, interfaces with Mathy to tackle mathy's large, sparse, action spaces.


```python
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
from mathy_core import MathTypeKeysMax
from mathy_envs import EnvRewards, MathyEnv, MathyEnvState
from mathy_envs.gym import MathyGymEnv

from mathy.fragile.env import DiscreteEnv
from mathy.fragile.models import DiscreteModel
from mathy.fragile.states import StatesEnv, StatesModel, StatesWalkers
from mathy.fragile.swarm import Swarm
from mathy.fragile.distributed_env import ParallelEnv


# Use multiprocessing to speed up the swarm
use_mp: bool = True
# Print the step-by-step output
verbose: bool = False
# The number of walkers to use in the swarm
n_walkers: int = 512
# The number of iterations to run the swarm for
max_iters: int = 100
```

### Action Selection

Fragile FMC defines a "Model" for doing action selection for the walkers in the swarm. Each walker in `n_walkers` needs to select actions, so we do it across large batches here.

To aid in navigating the large sparse action space, we'll use the action mask included in mathy observations (by default) to select only valid actions at each swarm step.


```python
class DiscreteMasked(DiscreteModel):
    def sample(
        self,
        batch_size: int,
        model_states: StatesModel,
        env_states: StatesEnv,
        walkers_states: StatesWalkers,
        **kwargs
    ) -> StatesModel:
        if env_states is not None:
            # Each state is a vstack([node_ids, mask]) and we only want the mask.
            masks = env_states.observs[:, -self.n_actions :]
            axis = 1
            # Select a random action using the mask to filter out invalid actions
            random_values = np.expand_dims(
                self.random_state.rand(masks.shape[1 - axis]), axis=axis
            )
            actions = (masks.cumsum(axis=axis) > random_values).argmax(axis=axis)
        else:
            actions = self.random_state.randint(0, self.n_actions, size=batch_size)
        return self.update_states_with_critic(
            actions=actions,
            model_states=model_states,
            batch_size=batch_size,
            **kwargs,
        )
```

### Planning Wrapper

Because FMC uses a swarm of many workers, it's vastly more efficient if you can interact with them in batches, similar to how we did above with action section. 

To support batch environments with stepping, etc, we'll implement a wrapper environment that supports the expected plangym interface, and creates an internal mathy environment. The class will also implement the `step_batch` method for stepping a batch of environments at once.


```python
import gymnasium as gym
from gymnasium import spaces


class PlanningEnvironment:
    """Fragile Environment for solving Mathy problems."""

    problem: Optional[str]

    @property
    def unwrapped(self) -> MathyGymEnv:
        return cast(MathyGymEnv, self._env.unwrapped)

    def __init__(
        self,
        name: str,
        environment: str = "poly",
        difficulty: str = "normal",
        problem: Optional[str] = None,
        max_steps: int = 64,
        **kwargs,
    ):
        self._env = gym.make(
            f"mathy-{environment}-{difficulty}-v0",
            invalid_action_response="terminal",
            env_problem=problem,
            mask_as_probabilities=True,
            **kwargs,
        )
        self.observation_space = spaces.Box(
            low=0,
            high=MathTypeKeysMax,
            shape=(256, 256, 1),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(self._env.unwrapped.action_size)
        self.problem = problem
        self.max_steps = max_steps
        self._env.reset()

    def get_state(self) -> np.ndarray:
        assert self.unwrapped.state is not None, "env required to get_state"
        return self.unwrapped.state.to_np(2048)

    def set_state(self, state: np.ndarray):
        assert self.unwrapped is not None, "env required to set_state"
        self.unwrapped.state = MathyEnvState.from_np(state)
        return state

    def step(
        self, action: int, state: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, Any, bool, Dict[str, object]]:
        assert self._env is not None, "env required to step"
        assert state is not None, "only works with state stepping"
        self.set_state(state)
        obs, reward, _, _, info = self._env.step(action)
        oob = not info.get("valid", False)
        new_state = self.get_state()
        return new_state, obs, reward, oob, info

    def step_batch(
        self,
        actions,
        states: Optional[Any] = None,
        n_repeat_action: Optional[Union[int, np.ndarray]] = None,
    ) -> tuple:
        data = [self.step(action, state) for action, state in zip(actions, states)]
        new_states, observs, rewards, terminals, infos = [], [], [], [], []
        for d in data:
            new_state, obs, _reward, end, info = d
            new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            infos.append(info)
        return new_states, observs, rewards, terminals, infos

    def reset(self, batch_size: int = 1):
        assert self._env is not None, "env required to reset"
        obs, info = self._env.reset()
        return self.get_state(), obs
```

### FMC Environment

To use the batch planning environment, we need to create a  Mathy environment that extends the discrete envrionment exposed by Fragile.

There's not too much special here, we instantiate the planning environment for use in the base class, and implement the `make_transition` function to set terminal states according to mathy_envs "done" property.



```python
class FMCEnvironment(DiscreteEnv):
    """Fragile FMC Environment for solving Mathy problems."""

    def __init__(
        self,
        name: str,
        environment: str = "poly",
        difficulty: str = "easy",
        problem: Optional[str] = None,
        max_steps: int = 64,
        **kwargs,
    ):
        self._env = PlanningEnvironment(
            name=name,
            environment=environment,
            difficulty=difficulty,
            problem=problem,
            max_steps=max_steps,
            **kwargs,
        )
        self._n_actions = self._env.action_space.n
        super(DiscreteEnv, self).__init__(
            states_shape=self._env.get_state().shape,
            observs_shape=self._env.observation_space.shape,
        )

    def make_transitions(
        self, states: np.ndarray, actions: np.ndarray, dt: Union[np.ndarray, int]
    ) -> Dict[str, np.ndarray]:
        new_states, observs, rewards, oobs, infos = self._env.step_batch(
            actions=actions, states=states
        )
        terminals = [inf.get("done", False) for inf in infos]
        data = {
            "states": np.array(new_states),
            "observs": np.array(observs),
            "rewards": np.array(rewards),
            "oobs": np.array(oobs),
            "terminals": np.array(terminals),
        }
        return data
```

## Swarm Solver

Now that we've setup a masked action selector and a batch-capable environment for planning with many walkers, we can put it all together and use the power of the Fractal Monte Carlo swarm to find a path to our desired solution.


```python
def swarm_solve(problem: str, max_steps: int = 256, silent: bool = False) -> None:
    def mathy_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance between two arrays."""
        return np.linalg.norm(x - y, axis=1)

    def env_callable():
        """Environment setup for solving the given problem."""
        return FMCEnvironment(
            name="mathy_v0",
            problem=problem,
            repeat_problem=True,
            max_steps=max_steps,
        )

    mathy_env: MathyEnv = env_callable()._env.unwrapped.mathy

    if use_mp:
        env_callable = ParallelEnv(env_callable=env_callable)

    swarm = Swarm(
        model=lambda env: DiscreteMasked(env=env),
        env=env_callable,
        reward_limit=EnvRewards.WIN,
        n_walkers=n_walkers,
        max_epochs=max_iters,
        reward_scale=1,
        distance_scale=3,
        distance_function=mathy_dist,
        show_pbar=False,
    )

    if not silent:
        print(f"Solving {problem} ...\n")
    swarm.run()

    if not silent:
        if swarm.walkers.best_reward > EnvRewards.WIN:
            last_state = MathyEnvState.from_np(swarm.walkers.states.best_state)
            mathy_env.print_history(last_state)
            print(f"Solved! {problem} = {last_state.agent.problem}")
        else:
            print("Failed to find a solution.")
        print(f"\nBest reward: {swarm.walkers.best_reward}\n\n")
```

## Evaluation

So, after all that work we can finally test and see how well the swarm is able to solve the problems we input. Let's give it a go!

> It's important to remember that the environment we've chosen only has a certain set of rules, so problems that rely on other rules to solve will not work here.

Let's recall which rules are available in the environment, and solve a few problems:


```python
env = FMCEnvironment(name="mathy_v0")
rules = "\n\t".join([e.name for e in env._env.unwrapped.mathy.rules])
print(f"Environment rules:\n\t{rules}\n")

swarm_solve("2x * x + 3j^7 + (1.9x^2 + -8y)")

swarm_solve("4x + 2y + 3j^7 + 1.9x + -8y")
```

    Environment rules:
    	Constant Arithmetic
    	Commutative Swap
    	Distributive Multiply
    	Distributive Factoring
    	Associative Group
    	Variable Multiplication
    	Restate Subtraction
    
    Solving 2x * x + 3j^7 + (1.9x^2 + -8y) ...
    
    initial-state(-1)         | 2x * x + 3j^7 + (1.9x^2 + -8y)
    variable multiplication(3) | 2x^(1 + 1) + 3j^7 + (1.9x^2 + -8y)
    associative group(19)     | 2x^(1 + 1) + 3j^7 + 1.9x^2 + -8y
    constant arithmetic(5)    | 2x^2 + 3j^7 + 1.9x^2 + -8y
    commutative swap(5)       | 3j^7 + 2x^2 + 1.9x^2 + -8y
    distributive factoring(11) | 3j^7 + (2 + 1.9) * x^2 + -8y
    constant arithmetic(7)    | 3j^7 + 3.9x^2 + -8y
    Solved! 2x * x + 3j^7 + (1.9x^2 + -8y) = 3j^7 + 3.9x^2 + -8y
    
    Best reward: 1.34333336353302
    
    
    Solving 4x + 2y + 3j^7 + 1.9x + -8y ...
    
    initial-state(-1)         | 4x + 2y + 3j^7 + 1.9x + -8y
    commutative swap(13)      | 4x + 2y + 1.9x + 3j^7 + -8y
    commutative swap(7)       | 4x + 1.9x + 2y + 3j^7 + -8y
    associative group(7)      | 4x + 1.9x + (2y + 3j^7) + -8y
    associative group(3)      | 4x + (1.9x + (2y + 3j^7)) + -8y
    distributive factoring(3) | (4 + 1.9) * x + (2y + 3j^7) + -8y
    commutative swap(15)      | (4 + 1.9) * x + -8y + (2y + 3j^7)
    constant arithmetic(1)    | 5.9x + -8y + (2y + 3j^7)
    distributive factoring(7) | 5.9x + (-8 + 2) * y + 3j^7
    constant arithmetic(5)    | 5.9x + -6y + 3j^7
    Solved! 4x + 2y + 3j^7 + 1.9x + -8y = 5.9x + -6y + 3j^7
    
    Best reward: 1.202222228050232
    
    


## Conclusion

If you're reading this, it means you either skipped ahead or you're an absolute legend! Either way, congrats! 🫠

I hope you now have a better understanding of how planning algorithms can integrate with Mathy to facilitate complex environment solving without trained models.


