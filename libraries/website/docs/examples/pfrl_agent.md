# PFRL Mathy Agent [![Open Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justindujardin/mathy/blob/master/libraries/website/docs/examples/pfrl_agent.ipynb)

> This notebook is built using [pfrl](https://github.com/pfnet/pfrl){target=\_blank} and [Mathy](https://mathy.ai).

Remember in Algebra how you had to combine "like terms" to simplify problems?

You'd see expressions like `60 + 2x^3 - 6x + x^3 + 17x` that have **5** total terms but only **4** "like terms".

That's because `2x^3` and `x^3` are like and `-6x` and `17x` are like, while `60` doesn't have any other terms that are like it.

Can we teach an agent to solve these kinds of problems step-by-step?

Let's give it a shot using [Mathy](https://mathy.ai) to generate math problems and [pfrl](https://github.com/pfnet/pfrl).



```
!pip install pfrl mathy_envs[gym]
```

    Collecting pfrl
    [?25l  Downloading https://files.pythonhosted.org/packages/93/cc/f26326d2a422d299cc686ae387bf1127f6ea11b2c2a85dae692eda0511f6/pfrl-0.1.0-py3-none-any.whl (149kB)
    [K     |████████████████████████████████| 153kB 5.7MB/s 
    [?25hCollecting mathy_envs[gym]
    [?25l  Downloading https://files.pythonhosted.org/packages/bb/a6/e838d117069d53cabfd505d26028948e48ee8f3213cfe94d371e8e8bf2ee/mathy_envs-0.9.3-py3-none-any.whl (43kB)
    [K     |████████████████████████████████| 51kB 6.2MB/s 
    [?25hRequirement already satisfied: numpy>=1.10.4 in /usr/local/lib/python3.6/dist-packages (from pfrl) (1.18.5)
    Requirement already satisfied: torch>=1.3.0 in /usr/local/lib/python3.6/dist-packages (from pfrl) (1.7.0+cu101)
    Requirement already satisfied: gym>=0.9.7 in /usr/local/lib/python3.6/dist-packages (from pfrl) (0.17.3)
    Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from pfrl) (7.0.0)
    Collecting pydantic>=1.0.0
    [?25l  Downloading https://files.pythonhosted.org/packages/0d/70/315a190f48b22e9a3918bc050af5ccd68c2d1db322c23f5f38af1313a20a/pydantic-1.7.2-cp36-cp36m-manylinux2014_x86_64.whl (9.2MB)
    [K     |████████████████████████████████| 9.2MB 26.7MB/s 
    [?25hCollecting colr
    [?25l  Downloading https://files.pythonhosted.org/packages/43/a9/75bcc155e0bf57062e77974a5aea724123de3becd69fca6f8572127c09a2/Colr-0.9.1.tar.gz (116kB)
    [K     |████████████████████████████████| 122kB 50.1MB/s 
    [?25hRequirement already satisfied: wasabi in /usr/local/lib/python3.6/dist-packages (from mathy_envs[gym]) (0.8.0)
    Collecting mathy-core>=0.8.2
    [?25l  Downloading https://files.pythonhosted.org/packages/d1/3a/ca15993c9eae67825845d501f357086f7cb0fb8ac0c32d8c372c040a50a3/mathy_core-0.8.2-py3-none-any.whl (69kB)
    [K     |████████████████████████████████| 71kB 7.6MB/s 
    [?25hRequirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from torch>=1.3.0->pfrl) (0.16.0)
    Requirement already satisfied: dataclasses in /usr/local/lib/python3.6/dist-packages (from torch>=1.3.0->pfrl) (0.7)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.6/dist-packages (from torch>=1.3.0->pfrl) (3.7.4.3)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from gym>=0.9.7->pfrl) (1.4.1)
    Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from gym>=0.9.7->pfrl) (1.5.0)
    Requirement already satisfied: cloudpickle<1.7.0,>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from gym>=0.9.7->pfrl) (1.3.0)
    Building wheels for collected packages: colr
      Building wheel for colr (setup.py) ... [?25l[?25hdone
      Created wheel for colr: filename=Colr-0.9.1-cp36-none-any.whl size=78233 sha256=6809961a455e732e3dc690775ccdf5c8ddfcc35e208cab12b1b883bba0cf11a5
      Stored in directory: /root/.cache/pip/wheels/76/e4/56/3db5b327cb8c9b4f877dd2841222b6496e394ea26ac20718b0
    Successfully built colr
    Installing collected packages: pfrl, pydantic, colr, mathy-core, mathy-envs
    Successfully installed colr-0.9.1 mathy-core-0.8.2 mathy-envs-0.9.3 pfrl-0.1.0 pydantic-1.7.2


### Verify The Environment

Before we write too much code, let's verify that we know the Mathy environment works and what kind of data we'll be working with.


```
import gym
from mathy_envs.gym import MathyGymEnv

env_name = f"mathy-poly-easy-v0"
env: MathyGymEnv = gym.make(env_name)  # type:ignore
# Set to 0 if you have a GPU
gpu = -1

print("observation space:", env.observation_space)
print("action space:", env.action_space)

obs = env.reset()
print(obs.tolist())
print(obs.min())
print(obs.max())
print(obs.std())
```

    observation space: Box(0.0, 1.0, (1027,), float32)
    action space: Discrete(768)
    [-3.5978141425591997e+18, 3.5978141425591997e+18, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05714285746216774, 0.20000000298023224, 1.0, 0.11428571492433548, 1.0, 0.20000000298023224, 1.0, 0.05714285746216774, 0.20000000298023224, 0.11428571492433548, 1.0, 0.20000000298023224, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5714285969734192, 0.0, 0.0, 0.7142857313156128, 0.0, 0.0, 0.5714285969734192, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    -3.5978141425591997e+18
    3.5978141425591997e+18
    1.587700204238583e+17


### Masked Action Space

As you probably noticed from the previous output, Mathy environments have quite large action spaces. 

The size of the action space is determined by the number of user-defined rules that the environment uses and the maximum sequence length that the environment will encode an observation of. Specifically, the action space has shape (num_rules, max_seq_len) and pads empty elements with 0.

In addition to being large, the action space often contains many invalid action choices. The envrionment exports a mask of valid actions as part of the observation to allow ignoring invalid actions during selection. Specifically, the last (num_rules * max_seq_len) elements of the observation are a binary (0/1) mask where 0 indicates the action is not valid in the current state.

We use the valid action mask exported by the environment to provide a custom action selector that extends PFRL's `DiscreteActionValue` class to mask the Q values so the agent can only select valid actions. This makes it *much* easier for the agent to find solutions early-on in training.


```
import torch
import torch.nn.functional as F
from pfrl.action_value import DiscreteActionValue
from torch.distributions.utils import lazy_property

class MaskedDiscreteActionValue(DiscreteActionValue):
    """Q-function output for masked discrete action space.

    Args:
        q_values (torch.Tensor):
            Array of Q values whose shape is (batchsize, n_actions)
    """

    def __init__(self, *, q_values, mask, q_values_formatter=lambda x: x):
        super().__init__(q_values=q_values, q_values_formatter=q_values_formatter)
        assert isinstance(q_values, torch.Tensor)
        self.mask = mask
        assert self.q_values.shape == self.mask.shape

    @lazy_property
    def greedy_actions(self):
        return self.masked_q.detach().argmax(axis=1).int()

    @lazy_property
    def masked_q(self):
        # Multiply by mask and then flip sign so that any remaining values
        # are greater than all masked values.
        return self.q_values.mul(self.mask).abs()

    @lazy_property
    def max(self):
        index = self.greedy_actions.long().unsqueeze(1)
        return self.masked_q.gather(dim=1, index=index).flatten()

    def evaluate_actions(self, actions):
        index = actions.long().unsqueeze(1)
        return self.masked_q.gather(dim=1, index=index).flatten()

    def compute_advantage(self, actions):
        return self.evaluate_actions(actions) - self.max

    def compute_double_advantage(self, actions, argmax_actions):
        return self.evaluate_actions(actions) - self.evaluate_actions(argmax_actions)

    def compute_expectation(self, beta):
        return torch.sum(F.softmax(beta * self.masked_q) * self.masked_q, dim=1)

    def __repr__(self):
        return "MaskedDiscreteActionValue greedy_actions:{} q_values:{}".format(
            self.greedy_actions.detach().cpu().np(),
            self.q_values_formatter(self.masked_q.detach().cpu().np()),
        )

    @property
    def params(self):
        return (self.masked_q,)

    def __getitem__(self, i):
        return MaskedDiscreteActionValue(
            q_values=self.q_values[i],
            q_values_formatter=self.q_values_formatter,
            mask=self.mask[i],
        )
```




```
# All ones for Q values given
q_values = torch.ones((1,512))
# Mask out all but 2 values
mask_values = torch.zeros((1,512))
mask_values[0][0] = 1.0
mask_values[0][12] = 1.0

head = MaskedDiscreteActionValue(q_values=q_values, mask=mask_values)

# Inspecting the masked_q property reveals only the masked elements are left
assert head.masked_q.sum() == 2.0
assert head.masked_q[0][0] == 1.0
assert head.masked_q[0][12] == 1.0

# All actions sampled are either 0 or 12
for i in range(100):
  assert head.greedy_actions in [0, 12]
```


```
import torch

class QFunction(torch.nn.Module):
    def __init__(self, obs_size: int, n_actions: int):
        super().__init__()
        self.n_actions = n_actions
        self.h_size = 128
        self.l1 = torch.nn.Linear(obs_size, self.h_size)
        self.l2 = torch.nn.Linear(self.h_size + obs_size, self.h_size)
        self.l3 = torch.nn.Linear(self.h_size + obs_size, self.h_size)
        self.l4 = torch.nn.Linear(self.h_size + obs_size, 64)
        self.l5 = torch.nn.Linear(64, n_actions)

    def forward(self, x):
        out = x
        out = torch.nn.functional.relu(self.l1(out))
        out = torch.nn.functional.relu(self.l2(torch.cat([out, x], -1)))
        out = torch.nn.functional.relu(self.l3(torch.cat([out, x], -1)))
        out = torch.nn.functional.relu(self.l4(torch.cat([out, x], -1)))
        out = self.l5(out)

        # The action mask is the last (n_action) values in the observation
        batch_mask = x[:, -self.n_actions :]
        assert batch_mask.shape == out.shape, "mask doesn't match output"
        return MaskedDiscreteActionValue(q_values=out, mask=batch_mask)

```


```

def make_agent(env: MathyGymEnv, gamma=0.9):
    def feature_extractor(observation):
        obs = torch.Tensor(observation).float()
        if gpu != -1:
            obs = obs.cuda()
        return obs

    obs_size = env.observation_space.low.size
    n_actions = env.action_space.n
    q_func = QFunction(obs_size, n_actions)
    optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)
    # Use epsilon-greedy for exploration
    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=0.7,
        end_epsilon=0.05,
        decay_steps=50000,
        random_action_func=env.action_space.sample,
    )
    replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

    # Now create an agent that will interact with the environment.
    _agent = pfrl.agents.DoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma,
        explorer,
        replay_start_size=1000,
        update_interval=4,
        target_update_interval=100,
        phi=feature_extractor,
        gpu=gpu,
    )
    return _agent

```


```
import logging
import numpy
from collections import deque
from typing import Any

class MyLogger(logging.Logger):
    last_msg: str
    eval_wins_window: deque
    eval_total_rewards_window: deque
    wins_window: deque
    total_rewards_window: deque

    def info(self, msg: Any, *args: Any, **kwargs: Any,) -> None:
        if not hasattr(self, "last_msg"):
            self.last_msg = ""
        # Training
        if msg == "outdir:%s step:%s episode:%s R:%s":
            if self.last_msg != msg:
                print("")
                self.wins_window = deque(maxlen=100)
                self.total_rewards_window = deque(maxlen=100)
            total_reward = args[-1]
            episode = args[-2]
            step = args[-3]
            self.wins_window.append(1.0 if total_reward > 0.0 else 0.0)
            self.total_rewards_window.append(total_reward)
            success_rate = (numpy.sum(self.wins_window)) / 100
            out = "\rTRAIN ep:{}\tmean:{:.2f}\tsuccess:{:.2f}".format(
                episode, numpy.mean(self.total_rewards_window), success_rate
            )
            print(out, end="")
        # Statistics
        elif msg == "statistics:%s":
            return
        # Evaluation
        elif msg == "evaluation episode %s length:%s R:%s":
            if self.last_msg != msg:
                print("")
                self.eval_wins_window = deque(maxlen=100)
                self.eval_total_rewards_window = deque(maxlen=100)
            total_reward = args[-1]
            episode = args[0]
            self.eval_wins_window.append(1.0 if total_reward > 0.0 else 0.0)
            self.eval_total_rewards_window.append(total_reward)
            mean_r = numpy.mean(self.eval_total_rewards_window)
            success_rate = numpy.sum(self.eval_wins_window) / 100
            out = "\rEVAL ep:{} \tmean R: {:.2f} \twin rate: {:.2f}".format(
                episode, mean_r, success_rate
            )
            print(out, end="")
        # Unknown
        else:
            return
        self.last_msg = msg

```


```
import pfrl

agent = make_agent(env)
outdir = f"training/poly_easy_ddqn"
print(f"==== Saving to: {outdir}")

pfrl.experiments.train_agent_with_evaluation(
    agent,
    env,
    steps=1_000_000,  # Train the agent for [n] steps
    eval_n_steps=None,  # We evaluate for episodes, not time
    eval_n_episodes=100,  # [n] episodes are sampled for each evaluation
    eval_max_episode_len=256,
    train_max_episode_len=256,  # Maximum length of each episode
    eval_interval=1000,
    successful_score=10.0,
    outdir=outdir,
    logger=MyLogger("mathy_pfrl"),
    use_tensorboard=True,
)
print("Finished.")
```

    ==== Saving to: training/poly_easy_ddqn
    
    TRAIN ep:65	mean:-0.88	success:0.10
    EVAL ep:99 	mean R: -1.22 	win rate: 0.00
    TRAIN ep:141	mean:-0.67	success:0.17
    EVAL ep:99 	mean R: -1.23 	win rate: 0.00
    TRAIN ep:214	mean:-0.52	success:0.21
    EVAL ep:99 	mean R: -1.01 	win rate: 0.08
    TRAIN ep:291	mean:-0.65	success:0.18
    EVAL ep:99 	mean R: -1.23 	win rate: 0.00
    TRAIN ep:361	mean:-1.00	success:0.07
    EVAL ep:99 	mean R: -1.23 	win rate: 0.00
    TRAIN ep:423	mean:-1.02	success:0.06
    EVAL ep:99 	mean R: -1.22 	win rate: 0.00
    TRAIN ep:459	mean:-0.57	success:0.10
