## Overview

Mathy uses machine learning (ML) to choose which [actions](/cas/rules/overview) to apply to which nodes in an expression tree in order to accomplish a [desired task](/envs/overview). Specifically, [Tensorflow 2.x](https://www.tensorflow.org/){target=\_blank} is used with the [Keras API](https://keras.io/){target=\_blank} to implement a [reinforcement learning platform](/envs/overview).

## Features

Mathy uses machine learning in a few ways, and has the following features:

- [Math Embeddings](/ml/embeddings) layer for processing sequences of sequences
- [Policy/Value model](/ml/policy_value_model) for estimating which actions to apply to which nodes
- [A3C agent](/ml/a3c) for online training of agents in a CPU-friendly setting
- [MCTS agent](/ml/zero) for batch training in a GPU-friendly setting

## Text Preprocessing

Mathy processes an input problem by parsing its text into a tree, converting that tree into a sequence features for each node in the tree, combining those features with the current environment state, and embedds them into a variable length sequence of fixed-dimension embeddings.

### Text to Trees

A problem text is [encoded into tokens](/cas/tokenizer), then [parsed into a tree](/cas/parser) that preserves the order of operations while removing parentheses and whitespace.
Consider the tokens and tree that result from the input: `-3 * (4 + 7)`

**Tokens**

`tokens:-3 * (4 + 7)`

**Tree**

`mathy:-3 * (4 + 7)`

Observe that the tree representation is more concise than the tokens array because it doesn't have nodes for hierarchical features like parentheses.

### Trees to Lists

Rather than try to feed [expression trees](/cas/parser) into a machine learning model, we [traverse them](/api/core/expressions/#to_list) to produce feature input sequences.

`features:-3 * (4 + 7)`

- The first row is the input token characters stripped of whitespace and parentheses.
- The second row is the sequence of floating point values for the tree, with each non-constant node represented by a mask value.
- The third row is the node type integer representing the class of the node in the tree.

!!! info "tree list ordering"

        You might have noticed the features from the previous tree are not expressed in the natural order that we might read them. As observed by [Lample and Charton](https://arxiv.org/pdf/1912.01412.pdf){target=\_blank} trees must be visited in an order that preserves the order-of-operations, so the model can pick up on the hierarchical features of the input.

        For this reason we visit trees in `pre` order for serialization.

### Lists to Observations

While feature lists may be directly passable to a ML model, they don't include any information about the state of the problem over time. To work with information over time, mathy agents draw extra information from the environment when building observations. This extra information includes:

- **Environment Problem Type**: environments all specify an [environment namespace](/api/env/#get_env_namespace) that is converted into a pair of [hashed string values](/api/state/#get_problem_hash) using different random seeds.
- **Episode Relative Time**: each observation is able to see a 0-1 floating point value that indicates how close the agent is to running out of moves.
- **[Current](/about/#r2d2) and [Historical](/about/#persistence-pays-off) RNN states**: observations include the recurrent neural network (RNN) state of the agent, and a historical average state from all the timesteps in the current episode.
- **Valid Action Mask**: mathy gives weighted estimates for each action at every node. If there are 5 possible actions, and 10 nodes in the tree, there are **up to** 50 possible actions to choose from. A same sized (e.g. 50) mask of 0/1 values are provided so that the model can mask out nodes with no valid actions when returning probability distributions.
