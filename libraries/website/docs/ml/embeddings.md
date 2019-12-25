# Overview

Mathy's embeddings model takes in a [window of observations](/api/state/#mathywindowobservation) and outputs a sequence of the same length with fixed-size learned embeddings for each token in the sequence.

## Model

The **Mathy** embeddings model is a stateful model that predicts over sequences. This complicates the process of collecting [observations](/api/state/#mathyobservation) to feed to the model, but allows richer input features than would be available from the simpler [state representation](/api/state/#mathyenvstate).

The model accepts an encoded sequence of tokens and values extracted from the current state's expression tree, and RNN state variables to use wit the recurrent processing layers.

`model:mathy.agents.embedding:mathy_embedding`

## Examples

### Observations to Embeddings

You can instantiate a model and produce untrained embeddings:

```python
{!./snippets/ml/embeddings_inference.py!}
```

### Access RNN states

The embeddings model is stateful and you can access the current recurrent network **hidden** and **cell** states.

```python
{!./snippets/ml/embeddings_rnn_state.py!}
```

!!! note "Hidden and Cell states"

    While the cell state is important to maintain while processing seqeuences, the hidden state is most often used for making predictions. This is because it is considered to contain the most useful representation of the RNN's memory.

## Env Features Model

Mathy has a conditionally enabled extended architecture that processes the `time` and `type` inputs when producing embeddings. In theory these features should be useful for multitask training, but in practice the model performs worse.

!!! warning "Note about performance"

    It's entirely possible that the extended architecture could be a viable improvement to the agents, and that I screwed something up that made it appear to perform worse. It's probably worth more investigation.

    If you have experience enough to spot why the env_features model performs poorly, or you are able to tinker it into a good state, please [open a PR]() so everyone can benefit from your fixes.

`model:mathy.agents.embedding:mathy_embedding_with_env_features`
