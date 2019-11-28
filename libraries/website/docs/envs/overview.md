Mathy includes a framework for building [reinforcement learning](/ml/reinforcement_learning) environments that transform math expressions using a set of user-defined actions.

There are a number of built-in environments aimed at simplifying algebra problems, and generous customization points for creating new ones.

### Episodes

Mathy agents interact with environments through sequences of ineteractions called episodes, which follow a standard RL episode lifecycle:

!!! info "Episode Pseudocode"

        1.  set **state** to an **initial state** from the **environment**
        2.  **while** **state** is not **terminal**
            - take an **action** and update **state**
        3.  **done**

### Extensions

Because algebra problems are only a tiny sliver of what can be represented using math expression trees, Mathy has customization points to allow altering or creating entirely new environments with little effort.

#### New Problems

Generating a new problem type while subclassing a base environment is probably the simplest way to create a custom challenge for the agent.

You can inherit from a base environment like [Poly Simplify](/envs/poly_simplify) which has win-conditions that require all the like-terms to be gone from an expression, and all complex terms be simplified. From there you can provide any valid input expression:

```Python
{!./snippets/envs/custom_problem_text.py!}
```

#### New Actions

Build your own tree transformation actions and use them with the built-in agents:

```Python
{!./snippets/envs/custom_actions.py!}
```

#### Custom Win Conditions

Environments can implement their own logic for win conditions, or inherit them from a base class:

```Python
{!./snippets/envs/custom_win_conditions.py!}
```

#### Custom Timestep Rewards

Specify which actions the agent should be rewarded for using and which it should be penalized for:

```Python
{!./snippets/envs/custom_timestep_rewards.py!}
```

#### Custom Episode Rewards

Specify (or calculate) custom floating point terminal reward values:

```Python
{!./snippets/envs/custom_episode_rewards.py!}
```
