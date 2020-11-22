Mathy includes a framework for building [reinforcement learning](/envs/overview) environments that transform math expressions using a set of user-defined actions.

Several built-in environments aim to simplify algebra problems and expose generous customization points for user-created ones.

### Episodes

Mathy agents interact with environments through sequences of interactions called episodes, which follow a standard RL episode lifecycle:

!!! info "Episode Pseudocode."

        1.  set **state** to an **initial state** from the **environment**
        2.  **while** **state** is not **terminal**
            - take an **action** and update **state**
        3.  **done**

### Extensions

Because algebra problems represent only a tiny sliver of the uses for math expression trees, Mathy has customization points to alter or create entirely new environments with little effort.

#### New Problems

Generating a new problem type while subclassing a base environment is probably the simplest way to create a custom challenge for the agent.

You can inherit from a base environment like [Poly Simplify](/envs/poly_simplify), which has win-conditions that require all the like-terms to be gone from an expression, and all complex terms be simplified. From there, you can provide any valid input expression:

```Python
{!./snippets/envs/custom_problem_text.py!}
```

#### New Actions

Build your tree transformation actions and use them with the built-in agents:

```Python
{!./snippets/envs/custom_actions.py!}
```

#### Custom Win Conditions

Environments can implement custom logic for win conditions or inherit them from a base class:

```Python
{!./snippets/envs/custom_win_conditions.py!}
```

#### Custom Timestep Rewards

Specify which actions to give the agent positive and negative rewards:

```Python
{!./snippets/envs/custom_timestep_rewards.py!}
```

#### Custom Episode Rewards

Specify (or calculate) custom floating-point episode rewards:

```Python
{!./snippets/envs/custom_episode_rewards.py!}
```

### Other Libraries

Mathy has support for alternative Reinforcement Learning libraries.

#### OpenAI Gym

Mathy has support [OpenAI gym](https://gym.openai.com/){target=\_blank} via a small wrapper.

You can import the `mathy_envs.gym` module separately to register the environments:

```python
{!./snippets/envs/openai_gym.py!}
```
