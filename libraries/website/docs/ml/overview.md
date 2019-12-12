## Overview

Mathy uses machine learning (or ML) to choose which [actions](/cas/rules/overview) to apply to which nodes in an expression tree in order to accomplish [a desired task](/envs/overview).

## Text Preprocessing

Mathy processes an input problem by parsing its text into a tree, converting that tree into a sequence features for each node in the tree, combining those features with the current environment state, and embedds them into a variable length sequence of fixed-dimension embeddings.

### Text to Trees

A problem text is [parsed into a tree](/cas/parser) that encodes the order of operations while removing parentheses and whitespace.
Consider the tree that results from the input: `-3 * (4 + 7)`

`mathy:-3 * (4 + 7)`

!!! info "Order of Operations"

        Remember that the order of operations normally puts higher priority on multiplication than addition, but in this
        case the addition of `4 + 7` has to be resolved first because it is explicitly grouped with parentheses.

### Trees to Lists

Rather than try to feed [expression trees](/cas/parser) into a machine learning model, we [traverse them](/api/core/expressions/#to_list) to produce feature input sequences.

Consider the tree from the previous example encoded in list form: `-3 * (4 + 7)`

`features:-3 * (4 + 7)`

- The first row is the input token characters stripped of whitespace and parentheses.
- The second row is the sequence of floating point values for the tree, with each non-constant node represented by a mask value.
- The third row is the node type integer representing the class of the node in the tree.

### Lists to Observations

While the feature lists from above may be directly passable to a ML model, they don't include any information about the state of the problem over time. To work with information over time, mathy agents draw extra information from the environment when building observations. This extra information includes:

- **Environment Problem Type**: environments all specify an [environment namespace](/api/env/#get_env_namespace) that is converted into a pair of [hashed string values](/api/state/#get_problem_hash) using different random seeds.
- **Episode Relative Time**: each observation is able to see a 0-1 floating point value that indicates how close the agent is to running out of moves.
- **[Current](/about/#r2d2) and [Historical](/about/#persistence-pays-off) RNN states**: observations include the recurrent neural network (RNN) state of the agent, and a historical average state from all the timesteps in the current episode.
- **Valid Action Mask**: mathy gives weighted estimates for each action at every node. If there are 5 possible actions, and 10 nodes in the tree, there are 50 possible actions to choose from. A same sized (e.g. 50) mask of 0/1 values are provided so that the model can mask out invalid logits when returning probability distributions.

### Observations to Embeddings

2. [embedded](/ml/math_embeddings) into variable length sequences of fixed size vectors that can be fed into a Sequence-to-Sequence ML model.

## Embeddings Model

Mathy's embeddings model takes in a [window of observations](/api/state/#mathywindowobservation) and outputs a sequence of the same length with fixed-size learned embeddings for each token in the sequence.

## Policy Value Model

Mathy's policy/value model takes in a [window of observations](/api/state/#mathywindowobservation) and outputs a weighted distribution over all the possible actions and value estimates for each observation.

## TODO

A set of feature lists for an expression is cool, but it represents only a single moment in time, so it's not a useful if your task is to transform a tree over time into a target state. To do that you need a concept of time, which is where Mathy's [reinforcement learning environments](/envs/overview) enter the equation.

Mathy environments start with an [initial state](/api/env/#get_initial_state) and the record the agent's actions, observations, and rewards after each timestep until the episode is complete.

When a Mathy agent wants to interact with the environment, it makes an [observation](/api/state/#mathyobservation) about the [current state](/api/state/#mathyenvstate) by looking at the environment and summarizing what it sees. State information that is available to build observations include, the problem text, the feature lists generated from it, metadata about the current episode, and a list of past observations from the same episode.
