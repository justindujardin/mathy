# History

## Why Mathy?

Mathy was an idea that came out of a tutoring session I had in college. I asked the tutor a dumb question, one that I should have known the answer to from highschool Algebra, and they choked down a look of disgust as they explained it to me.

It struck me that the "safe place to learn" which academia sought to provide didn't often feel safe. I felt ashamed and embarassed because I was missing something foundational for the math I needed to do, and I felt that I couldn't get the help I needed without facing ridicule.

I knew there had to be a better way to make foundational knowledge about math available to anyone that wanted it. I thought to myself, I know how to write software; I can build it! Some ten years later, here we are, trying to provide a free math tutor to the world.

Will you help?

## Research

A lot of research has informed the design of Mathy's machine learning model and agents.

### [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go){target=\_blank}

The work done by [Silver et al.](https://arxiv.org/pdf/1712.01815.pdf) demonstrated the power of taking a principled approach to solving previously intractable problems. It inspired Mathy to use Neural Networks and reinforcement learning to pick combinations of rules to transform the tree with. This work lead directly to the removal of a layer of "composite" rules in Mathy for combining like terms and simplifying expressions.

### [A3C](https://arxiv.org/pdf/1602.01783){target=\_blank}

The work done by [Mnih et al.](https://arxiv.org/pdf/1602.01783) in "Asynchronous Methods for Deep Reinforcement Learning" describes a CPU-friendly RL algorithm that can be trained for some tasks very rapidly. The `a3c` agent is based on this [tensorflow 2.0 implementation of A3C](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296).

### [Persistence Pays Off](https://arxiv.org/pdf/1810.04437.pdf){target=\_blank}

The work done by [Salton and Kelleher](https://arxiv.org/pdf/1810.04437.pdf) in "Persistence pays off: Paying Attention to What the LSTM Gating Mechanism Persists" observes that LSTMs create a problem when they remove information from their states while processing a sequence, namely that the information is no longer available to future steps in the sequence. They suggest this makes it hard to identify long-term dependencies, and to address it they store and average the RNN states in recent history, then combine it with others input to your model. Mathy tracks the RNN state and RNN history for all observations, and incorporates them in the embeddings layer.

### [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX){target=\_blank}

The work done by [Kapturowski et al.](https://openreview.net/pdf?id=r1lyTjAqYX){target=\_blank} in "Recurrent Experience Replay in Distributed Reinforcement Learning" shows that storing and properly initializing experience replay data can be crucial to identifying long-term dependencies in disitributed reinforcement learning. Mathy uses stored RNN states when training all sequences and optionally performs burn-in steps.

### [UNREAL](https://deepmind.com/blog/article/reinforcement-learning-unsupervised-auxiliary-tasks){target=\_blank}

The work done by [Jaderberg et al.](https://arxiv.org/pdf/1611.05397.pdf){target=\_blank} explores the use of unsupervised auxiliary tasks to encourage RL agents to converge at a much quicker rate than they otherwise would. They find that using shared network weights to predict different things can help agents learned shared representations that both can benefit from. Mathy uses an auxiliary task for controlling the grouping of like terms in an expression, inspired by this work.

## Open Source

The scope of Mathy is huge, and there are a few key contributions that have come from Open Source software that deserve special recognition.

### [Silverlight Graphing Calculator](https://code.msdn.microsoft.com/Silverlight-Graphing-fb30536e/){target=\_blank}

When I began researching how to transform text into expression trees I came across a graphing calculator written in C#. The original parser and tokenizer came from Bob Brown at Microsoft, and Mathy still uses a modified version of this system today. The clear and concise tokenizer/parser implementation have been endlessly helpful as Mathy has been ported to various languages.

??? note "Apache License, Version 2.0"
Copyright 2004 Microsoft Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

### [AlphaZero General](https://github.com/suragnair/alpha-zero-general){target=\_blank}

After learning about the success of [MCTS combined with Neural Networks](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go){target=\_blank} I was convinced that an alternative to custom written heuristics was available to transform trees in complex ways. It was at this time that I came across an AlphaGo implementation called [Alpha Zero General](https://github.com/suragnair/alpha-zero-general){target=\_blank} that I adapted to single-agent Mathy environments. While the code has changed quite a bit, you can continue see parts of AZG in the MathyEnv and MCTS classes. The `zero` agent is based on this approach.

!!! warning "Missing Functionality"

        At one point during an aggressive refactor and simplification I removed the self-play competition from the `zero` agent's training regimen. Later, I saw a comment from DeepMind about how self-play is critical to continual learning because it creates a competetive challenge that is never too hard or too easy.

        This code is relatively straight-forward and should be revived for optimal performance.

??? note "MIT License"

    Copyright (c) 2018 Surag Nair

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

### [TRFL](https://github.com/deepmind/trfl){target=\_blank}

Writing RL algorithms is difficult. The math is tricky, and when you mess it up your algorithm doesn't usually fail, it just sucks. DeepMind provides Tensorflow operations for many advanced RL losses. Mathy uses an inlined (Tensorflow 2.0 compatible) TRFL library for computing Policy Value network losses.

??? note "Apache License, Version 2.0"
Copyright 2018 The trfl Authors. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
    See the License for the specific language governing permissions and
    limitations under the License.
