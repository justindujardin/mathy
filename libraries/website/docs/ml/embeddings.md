# Overview

Mathy's embeddings model takes in a [window of observations](/api/state/#mathywindowobservation) and outputs a sequence of the same length with fixed-size learned embeddings for each token in the sequence.

## Model

The **Mathy** embeddings model is a stateful model that predicts over sequences. This complicates the process of collecting [observations](/api/state/#mathyobservation) to feed to the model, but allows richer input features than would be available from the simpler [state representation](/api/state/#mathyenvstate).

The model accepts an encoded sequence of tokens and values extracted from the current state's expression tree, and RNN state variables to use wit the recurrent processing layers.

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
