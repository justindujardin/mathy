# mathy.contrastive
Implements a contrastive pretraining CLI app for producing math expression
vector representations from character inputs
## add_contrastive_loss
```python
add_contrastive_loss(
    hidden: tensorflow.python.framework.ops.Tensor,
    hidden_norm: bool = True,
    temperature = 0.1,
    weights = 1.0,
)
```
Compute NT-Xent contrastive loss.

Copyright 2020 The SimCLR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific simclr governing permissions and
limitations under the License.
==============================================================================

Args:
hidden: hidden vector (`Tensor`) of shape (bsz, dim).
hidden_norm: whether or not to use normalization on the hidden vector.
temperature: a `floating` number for temperature scaling.
weights: a weighting number or vector.
Returns:
A loss scalar.
The logits for contrastive prediction task.
The labels for contrastive prediction task.

## build_projection_network
```python
build_projection_network(
    input_shape: Tuple[int, ...],
) -> tensorflow.python.keras.engine.sequential.Sequential
```
Projection model for building a richer representation of output vectors for the
contrastive loss task. The paper claims this greatly improved performance.
## decode_text
```python
decode_text(tokens:tensorflow.python.framework.ops.Tensor) -> str
```
Decode a list of integer tensors to produce a string
## encode_text
```python
encode_text(
    text: str,
    pad_length: int = 128,
    include_batch: bool = False,
) -> tensorflow.python.framework.ops.Tensor
```
Encode text into a list of indices in the vocabulary
## get_trainer
```python
get_trainer(
    folder: str,
    quiet: bool = False,
    training: bool = True,
    profile: bool = False,
) -> mathy.contrastive.ContrastiveModelTrainer
```
Create and return a trainer, optionally loading an existing model
## swish
```python
swish(x)
```
Swish activation function: https://arxiv.org/pdf/1710.05941.pdf
