# Machine Learning Solver [![Open Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justindujardin/mathy/blob/master/website/docs/examples/mathy_ppo.ipynb)

> This notebook is built using [mathy_envs](https://envs.mathy.ai) and a modified version of [@nikhilbarhate99](https://github.com/nikhilbarhate99)'s wonderful [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch) script.

While working with math problems using heuristics is interpretable and reliable, it can be a large engineering task to design combinations of rules and heuristics for handling all the various tree forms that user input questions might take.

Rather than invest engineering time into writing heuristics, we can use machine learning algorithms to train a model that can select which actions to take in order to find an optimal path to a solution. Not only is this more robust than random action selections, but it will make solving many types of problems trivial once we get going.

Let's look together at how [mathy_envs](https://envs.mathy.ai) can be used with Proximal Policy Optimization (PPO) in PyTorch to train a problem solving model that can then be used to demonstrate solving problems step-by-step. 



```python
!pip install mathy_envs>=0.12.1 torch
```

    
    [notice] A new release of pip is available: 23.3.1 -> 23.3.2
    [notice] To update, run: pip install --upgrade pip


## Overview

Before we get started, let's review what mathy envs are and how they work. Mathy envs are reinforcement learning environments for maniuplating math trees with a rules system.

1. Each mathy_envs environment generates math problem texts and determines if the current expression is "solved" or not
2. Users/Models interact with the environments by playing "episodes" where they solve problems given a set of rules and environment-specific logic
3. Depending on the context the outputs are either used as inputs to a training model, or as an output demonstration for an end-user

We're going to use reinforcement learning to train a model that is capable of solving problems generated by the mathy_envs library.

Specifically we choose the `PolySimplify` environment which generates controllably difficult polynomial simplification problems, and implements logic to determine when they're solved. 

For the machine learning portion, we choose the Proximal Policy Optimization algorithm, which is an online learning algorithm.

Before we get into the machine learning parts, let's quickly get a taste for the basics of our environments. 

Mathy envs implements a base environment interface, and that's wrapped in a set of classes exposed for gym/gymnasium libraries.


```python
import gymnasium as gym
import numpy as np
import torch
from mathy_envs import MathyEnv
from mathy_envs.gym import MathyGymEnv

# Environment difficulty level (options: 'easy', 'normal', 'hard')
env_difficulty = "easy"

# Environment names to train on, based on environment types and difficulty
env_types = [
    "poly",
    # "poly-blockers",
    # "poly-combine",
    # "poly-commute",
    # "poly-grouping",
    # "poly-like-terms-haystack",
    # "binomial",
    # "complex",
]
env_names = [f"mathy-{t}-{env_difficulty}-v0" for t in env_types]

env: MathyGymEnv = gym.make(env_names[0])
base_env: MathyEnv = env.unwrapped.mathy
print(f"Environment: {base_env.get_env_namespace()}")
print(f"Num Actions: {base_env.action_size}")
print(f"Rules      : {[e.name for e in base_env.rules]}")
```

    Environment: mathy.polynomials.simplify
    Num Actions: 896
    Rules      : ['Constant Arithmetic', 'Commutative Swap', 'Distributive Multiply', 'Distributive Factoring', 'Associative Group', 'Variable Multiplication', 'Restate Subtraction']


## Proximal Policy Optimization


Proximal Policy Optimization (PPO) is a reinforcement learning approach that strikes a balance between the simplicity of implementation and sample efficiency. Developed by John Schulman and his colleagues at OpenAI, PPO is designed to be more stable and reliable than earlier policy gradient methods, thanks to its novel objective function that moderates the policy updates.

PPO uses a buffer of trajectories called a rollout buffer to store key elements like states, actions, and rewards generated by interacting with the environments. These buffered trajectories are used when making updates to the actor critic neural networks during training.

We'll use PPO here to train an agent that is able to solve polynomial simplification problems in a step-by-step manner.

First, let's set some variables that can be changed later for experimentation.


```python
# Learning rate for the actor network
lr_actor = 0.0003

# Learning rate for the critic network
lr_critic = 0.001

# Discount factor for future rewards
gamma = 0.99

# Number of epochs to update the policy
K_epochs = 80

# Clip parameter for PPO, used in policy update
eps_clip = 0.2

# Random seed setting (0 = no random seed)
random_seed = 1337

# Device to run the training on (CPU or CUDA)
device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")

# Dimension of the hidden layer in the critic network
critic_hidden_dim = 64

# Where to save the model
checkpoint_path = "ppo.pth"

# Whether or not to use masked action selection. This makes the problems significantly easier when
# true because the action space is sparse with most possible actions being invalid. When false, the
# agent must learn to avoid invalid actions itself, making the problems much more challenging given
# the action space on the order of hundreds or thousands of possible actions for each state.
use_masked_actions = True
```

### Rollout Buffer

The Rollout Buffer in PPO stores the trajectory of the agent during its interaction with the environment over a single policy iteration. This includes actions, states, rewards, log probabilities of the actions under the current policy, and state values. When the policy is updated, the buffer is cleared.


```python
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
```

### Actor-Critic Module

The Actor-Critic module in PPO is the core of the policy learning mechanism, and has two key components:

- the Actor, which is responsible for choosing actions based on the current state.
- the Critic, which evaluates actor actions by estimating the value of the state. This is used to help steer the actor in the direction of higher value actions.

The dual structure allows for more efficient and stable learning by combining the strengths of both policy-based and value-based approaches in reinforcement learning.


```python
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.device = device
        self.action_dim = action_dim

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        )

    def act(self, state):
        action_probs = self.actor(state)
        if use_masked_actions:
            mask = state[-action_probs.shape[0] :]
            action_probs = action_probs * mask
            action_probs = action_probs / torch.sum(action_probs)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
```

### Algorithm

The PPO class encapsulates the Proximal Policy Optimization algorithm, a policy gradient method for reinforcement learning. It manages the training loop, including action selection, policy updating, and handling the experience buffer.

The class initializes two ActorCritic models: one for the current policy and another as a reference to the old policy. This structure is crucial for implementing PPO's clipped surrogate objective function, which moderates the policy updates for stability.

The class also includes methods for saving and loading model checkpoints.


```python
import torch
import torch.nn as nn


class PPO:
    def __init__(self, state_dim: int, action_dim: int):
        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(self.device)
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(self.device)
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(self.device)
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_old.load_state_dict(checkpoint)
        self.policy.load_state_dict(checkpoint)
```

## Training

The training loop is the main part of PPO. It makes the agent better by having it interact with its environment and learn from what happens. 

The loop includes picking actions with the current policy, getting rewards and states, and updating the policy regularly.

During training, we print updates about how many episodes we've finished, how many steps we've taken, and the average reward the agent is getting.

Once training is done, the improved model can make better decisions, leading to solutions that are closer to the best possible ones.



```python
def train(checkpoint_path: str, max_steps: int = 1_000_000):
    print(
        f"Device set to: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Training environments: {', '.join(env_names)}")

    max_ep_len = 50  # Max timesteps in one episode
    print_freq = 20_000  # Frequency for printing average reward
    save_model_freq = int(1e5)  # Model saving frequency
    update_timestep = max_ep_len * 4  # update policy every n timesteps

    envs = [
        gym.make(name, invalid_action_response="raise", verbose=False)
        for name in env_names
    ]
    env = envs[0]  # Select an environment

    # Initialize the PPO agent
    ppo_agent = PPO(env.observation_space.shape[0], env.action_space.n)

    # Training variables
    time_step = 0
    i_episode = 0
    total_reward = 0

    # Training loop
    while time_step <= max_steps:
        state, _ = env.reset()
        episode_reward = 0

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            episode_reward += reward

            if time_step % update_timestep == 0:
                ppo_agent.update()

            # Print average reward
            if time_step % print_freq == 0:
                avg_reward = total_reward / i_episode if i_episode > 0 else 0
                print(
                    f"Episode: {i_episode} \t Timestep: {time_step} \t Average Reward: {avg_reward:.2f}"
                )

            # Save model
            if time_step % save_model_freq == 0:
                print(f"Saving model at timestep {time_step}")
                ppo_agent.save(checkpoint_path)

            if done:
                break

        total_reward += episode_reward
        i_episode += 1

    print("Training completed.")
```

Now we're ready to train out model. If you want to train a fully-capable agent in this environment, you might want to train for 1 or more million steps with the default argument value, but that can take an hour or more. 

For our purposes a few hundred thousand steps is a good number because it should only take a few minutes and you can see the agent start to learn how to solve problems in that time.

Any reward values over **0.0** almost always indicate a correct solution within the number of steps allowed by the environment. Perfect scores are generally around **~1.5** for most environments and max out at about **2.0** for others that only take a few steps.


```python
train(checkpoint_path, 300_000)
```

    Device set to: NVIDIA GeForce RTX 3090
    Training environments: mathy-poly-easy-v0
    Episode: 1672 	 Timestep: 20000 	 Average Reward: -1.25
    Episode: 3752 	 Timestep: 40000 	 Average Reward: -0.54
    Episode: 6370 	 Timestep: 60000 	 Average Reward: 0.02
    Episode: 9146 	 Timestep: 80000 	 Average Reward: 0.34
    Episode: 12116 	 Timestep: 100000 	 Average Reward: 0.54
    Saving model at timestep 100000
    Episode: 15078 	 Timestep: 120000 	 Average Reward: 0.66
    Episode: 18083 	 Timestep: 140000 	 Average Reward: 0.75
    Episode: 21243 	 Timestep: 160000 	 Average Reward: 0.81
    Episode: 24630 	 Timestep: 180000 	 Average Reward: 0.88
    Episode: 27946 	 Timestep: 200000 	 Average Reward: 0.92
    Saving model at timestep 200000
    Episode: 31427 	 Timestep: 220000 	 Average Reward: 0.96
    Episode: 34634 	 Timestep: 240000 	 Average Reward: 0.99
    Episode: 38175 	 Timestep: 260000 	 Average Reward: 1.02
    Episode: 41833 	 Timestep: 280000 	 Average Reward: 1.05
    Episode: 45453 	 Timestep: 300000 	 Average Reward: 1.07
    Saving model at timestep 300000
    Training completed.


## Evaluation

The test function evaluates the performance of our trained PPO agent. It loads the agent's model from a saved checkpoint and runs it through multiple episodes in different environments. In each episode, the agent makes decisions based on its learned policy, and we track the rewards it earns. The main goal is to see how well the agent performs in these test scenarios, indicated by the total rewards it accumulates across episodes. This testing phase is crucial as it gives us a clear picture of the effectiveness of our training and the agent's ability to handle various challenges within the environments. The average reward per episode, calculated at the end, serves as a key metric to assess the agent's performance.


```python
def test(checkpoint_path: str):
    envs = [
        gym.make(name, invalid_action_response="raise", verbose=True)
        for name in env_names
    ]
    assert len(envs) > 0, "No environments found"
    env = envs[0]
    ppo_agent = PPO(env.observation_space.shape[0], env.action_space.n)
    
    print(f"\nloading network from : {checkpoint_path}\n", flush=True)
    ppo_agent.load(checkpoint_path)

    total_test_episodes = 10  # total num of testing episodes
    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        env = np.random.choice(envs)
        ep_reward = 0
        state, _ = env.reset()
        done = False

        while not done:
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        test_running_reward += ep_reward
        emoji = "✅" if ep_reward >= 0.0 else "🔴"
        print(f"[{ep}]{emoji} Reward: {round(ep_reward, 2)}")
        ep_reward = 0

    avg_test_reward = test_running_reward / total_test_episodes
    print(f"Average test reward: {round(avg_test_reward, 2)}")
```

Having trained a model and written the evalutaion, we can finally see the results of our hard work. Our tiny model (< 1MB) is able to solve our polynomial simplification problems somewhat consistently. 

With more training the given agent config can reach near perfect accuracy on this task.


```python
test(checkpoint_path)
```

    006 | -- cs -- df ag -- -- | 16 | 0.0   | initial-state(-1)         | (6j^2 + 2j^2 + 2q + 5r^3)
    
    loading network from : ppo.pth
    
    005 | -- cs -- -- ag -- -- | 16 | 0.0   | initial-state(-1)         | 6y + 8m^2 + 12y + 10m^2
    007 | -- cs -- df ag -- -- | 15 | -0.01 | commutative swap(9)       | 6y + 12y + 8m^2 + 10m^2
    008 | ca cs dm df ag -- -- | 14 | 0.01  | distributive factoring(3) | (6 + 12) * y + 8m^2 + 10m^2
    004 | -- cs -- df ag -- -- | 13 | 0.01  | constant arithmetic(1)    | 18y + 8m^2 + 10m^2
    004 | ca cs dm -- -- -- -- | 12 | 0.01  | distributive factoring(9) | 18y + (8 + 10) * ^2
    001 | -- cs -- -- -- -- -- | 11 | 1.4   | constant arithmetic(5)    | 18y + 18^2
    [1]✅ Reward: 1.42
    005 | -- cs -- -- ag -- -- | 16 | 0.0   | initial-state(-1)         | (3c^4 + 5o^3 + c^4) + 8c^2
    006 | -- cs -- df ag -- -- | 15 | -0.01 | commutative swap(11)      | 3c^4 + c^4 + 5o^3 + 8c^2
    006 | ca cs dm -- ag -- -- | 14 | 0.01  | distributive factoring(5) | (3 + 1) * c^4 + 5o^3 + 8c^2
    003 | -- cs -- -- ag -- -- | 13 | 1.7   | constant arithmetic(1)    | 4c^4 + 5o^3 + 8c^2
    [2]✅ Reward: 1.67
    007 | -- cs -- df ag -- -- | 16 | 0.0   | initial-state(-1)         | 7o^2 + (o^2 + g + 9g)
    007 | ca cs dm df ag -- -- | 15 | 0.01  | distributive factoring(5) | (7 + 1) * o^2 + (g + 9g)
    008 | ca cs dm -- -- -- -- | 14 | 0.01  | distributive factoring(9) | (7 + 1) * o^2 + (1 + 9) * g
    005 | ca cs dm -- -- -- -- | 13 | 0.01  | constant arithmetic(1)    | 8o^2 + (1 + 9) * g
    001 | -- cs -- -- -- -- -- | 12 | 1.5   | constant arithmetic(7)    | 8o^2 + 10g
    [3]✅ Reward: 1.53
    005 | -- cs -- -- ag -- -- | 16 | 0.0   | initial-state(-1)         | 7v + (10v^4 + v) + 7v^4
    007 | -- cs -- df ag -- -- | 15 | -0.01 | commutative swap(9)       | 7v + (v + 10v^4) + 7v^4
    007 | ca cs dm df ag -- -- | 14 | 0.01  | distributive factoring(11) | 7v + v + (10 + 7) * v^4
    004 | -- cs -- df ag -- -- | 13 | 0.01  | constant arithmetic(7)    | 7v + v + 17v^4
    005 | ca cs dm -- -- -- -- | 12 | 0.01  | distributive factoring(3) | (7 + 1) * v + 17v^4
    001 | -- cs -- -- -- -- -- | 11 | 1.4   | constant arithmetic(1)    | 8v + 17v^4
    [4]✅ Reward: 1.42
    006 | -- cs -- df ag -- -- | 16 | 0.0   | initial-state(-1)         | (9u^3 + 10r + 3r + 8u^3)
    007 | ca cs dm -- ag -- -- | 15 | 0.01  | distributive factoring(9) | 9u^3 + (10 + 3) * r + 8u^3
    008 | ca cs dm df ag -- -- | 14 | -0.01 | commutative swap(5)       | (10 + 3) * r + 9u^3 + 8u^3
    004 | -- cs -- df ag -- -- | 13 | 0.01  | constant arithmetic(1)    | 13r + 9u^3 + 8u^3
    004 | ca cs dm -- -- -- -- | 12 | 0.01  | distributive factoring(9) | 13r + (9 + 8) * u^3
    001 | -- cs -- -- -- -- -- | 11 | 1.4   | constant arithmetic(5)    | 13r + 17u^3
    [5]✅ Reward: 1.42
    003 | -- cs -- -- ag -- -- | 16 | 0.0   | initial-state(-1)         | (5p^3 + 2y + 8p^3)
    004 | -- cs -- df ag -- -- | 15 | -0.01 | commutative swap(5)       | 2y + 5p^3 + 8p^3
    004 | ca cs dm -- -- -- -- | 14 | 0.01  | distributive factoring(9) | 2y + (5 + 8) * p^3
    001 | -- cs -- -- -- -- -- | 13 | 1.7   | constant arithmetic(5)    | 2y + 13p^3
    [6]✅ Reward: 1.67
    009 | -- cs -- -- ag -- -- | 20 | 0.0   | initial-state(-1)         | (4a + 4f + 5f^2 + 8a + 12z + 7f)
    009 | -- cs -- -- ag -- -- | 19 | -0.01 | commutative swap(7)       | 4a + 5f^2 + 4f + 8a + 12z + 7f
    009 | -- cs -- -- ag -- -- | 18 | -0.01 | commutative swap(9)       | 4a + 4f + 5f^2 + 8a + 12z + 7f
    009 | -- cs -- -- ag -- -- | 17 | -0.04 | commutative swap(7)       | 4a + 5f^2 + 4f + 8a + 12z + 7f
    009 | -- cs -- -- ag -- -- | 16 | -0.04 | commutative swap(9)       | 4a + 4f + 5f^2 + 8a + 12z + 7f
    009 | -- cs -- -- ag -- -- | 15 | -0.06 | commutative swap(7)       | 4a + 5f^2 + 4f + 8a + 12z + 7f
    009 | -- cs -- -- ag -- -- | 14 | -0.01 | associative group(9)      | 4a + 5f^2 + (4f + 8a) + 12z + 7f
    009 | -- cs -- -- ag -- -- | 13 | -0.01 | associative group(9)      | 4a + 5f^2 + (4f + 8a + 12z) + 7f
    009 | -- cs -- -- ag -- -- | 12 | -0.01 | associative group(9)      | 4a + 5f^2 + (4f + 8a + 12z + 7f)
    009 | -- cs -- -- ag -- -- | 11 | -0.01 | commutative swap(9)       | 4a + (4f + 8a + 12z + 7f) + 5f^2
    009 | -- cs -- -- ag -- -- | 10 | -0.01 | commutative swap(11)      | 4a + (4f + 12z + 8a + 7f) + 5f^2
    009 | -- cs -- -- ag -- -- | 09 | -0.04 | commutative swap(11)      | 4a + (4f + 8a + 12z + 7f) + 5f^2
    009 | -- cs -- -- ag -- -- | 08 | -0.04 | commutative swap(11)      | 4a + (4f + 12z + 8a + 7f) + 5f^2
    009 | -- cs -- -- ag -- -- | 07 | -0.06 | commutative swap(11)      | 4a + (4f + 8a + 12z + 7f) + 5f^2
    009 | -- cs -- -- ag -- -- | 06 | -0.01 | commutative swap(15)      | 4a + (4f + 8a + 7f + 12z) + 5f^2
    010 | -- cs -- df ag -- -- | 05 | -0.01 | commutative swap(11)      | 4a + (4f + 7f + 8a + 12z) + 5f^2
    010 | -- cs -- df ag -- -- | 04 | -0.01 | commutative swap(3)       | 4f + 7f + 8a + 12z + 4a + 5f^2
    011 | ca cs dm -- ag -- -- | 03 | 0.01  | distributive factoring(3) | (4 + 7) * f + 8a + 12z + 4a + 5f^2
    007 | -- cs -- -- ag -- -- | 02 | 0.01  | constant arithmetic(1)    | 11f + 8a + 12z + 4a + 5f^2
    007 | -- cs -- -- ag -- -- | 01 | -0.01 | associative group(3)      | 11f + (8a + 12z) + 4a + 5f^2
    007 | -- cs -- -- ag -- -- | 00 | -1.0  | associative group(3)      | 11f + (8a + 12z + 4a) + 5f^2
    [7]🔴 Reward: -1.37
    002 | -- cs -- df -- -- -- | 08 | 0.0   | initial-state(-1)         | (12k^4 + 4k^4)
    003 | ca cs dm -- -- -- -- | 07 | 0.01  | distributive factoring(5) | (12 + 4) * k^4
    002 | -- cs -- df -- -- -- | 06 | -0.01 | distributive multiply(3)  | 12k^4 + 4k^4
    003 | ca cs dm -- -- -- -- | 05 | -0.04 | distributive factoring(5) | (12 + 4) * k^4
    000 | -- -- -- -- -- -- -- | 04 | 1.2   | constant arithmetic(1)    | 16k^4
    [8]✅ Reward: 1.21
    005 | -- cs -- -- ag -- -- | 16 | 0.0   | initial-state(-1)         | 12x + (8x^2 + 1x + x^2)
    007 | -- cs -- df ag -- -- | 15 | -0.01 | commutative swap(9)       | 12x + (1x + 8x^2 + x^2)
    008 | ca cs dm df ag -- -- | 14 | 0.01  | distributive factoring(3) | (12 + 1) * x + (8x^2 + x^2)
    004 | -- cs -- df ag -- -- | 13 | 0.01  | constant arithmetic(1)    | 13x + (8x^2 + x^2)
    004 | ca cs dm -- -- -- -- | 12 | 0.01  | distributive factoring(9) | 13x + (8 + 1) * x^2
    001 | -- cs -- -- -- -- -- | 11 | 1.4   | constant arithmetic(5)    | 13x + 9x^2
    [9]✅ Reward: 1.42
    006 | -- cs -- df ag -- -- | 12 | 0.0   | initial-state(-1)         | c + (8c + 6c^2 + 6o)
    005 | -- cs -- -- ag -- -- | 11 | -0.01 | commutative swap(1)       | 8c + 6c^2 + 6o + c
    005 | -- cs -- -- ag -- -- | 10 | -0.01 | commutative swap(9)       | 8c + 6o + 6c^2 + c
    005 | -- cs -- -- ag -- -- | 09 | -0.04 | commutative swap(7)       | 8c + 6c^2 + 6o + c
    005 | -- cs -- -- ag -- -- | 08 | -0.01 | associative group(9)      | 8c + 6c^2 + (6o + c)
    005 | -- cs -- -- ag -- -- | 07 | -0.01 | commutative swap(9)       | 8c + (6o + c) + 6c^2
    006 | -- cs -- df ag -- -- | 06 | -0.01 | commutative swap(7)       | 8c + (c + 6o) + 6c^2
    007 | ca cs dm -- ag -- -- | 05 | 0.01  | distributive factoring(3) | (8 + 1) * c + 6o + 6c^2
    003 | -- cs -- -- ag -- -- | 04 | 1.1   | constant arithmetic(1)    | 9c + 6o + 6c^2
    [10]✅ Reward: 1.04
    Average test reward: 1.14
