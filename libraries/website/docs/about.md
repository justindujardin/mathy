## Research

A lot of research has informed the design of Mathy's machine learning model and agents.

It's because of the insights from this research that Mathy works at all today.

In fact, Mathy's models perform well-enough to solve some types of problems reliably!

That said, I'm not a machine learning researcher and it's entirely possible that I've messed up on implementing some of the research listed below.

My apologies in advance to any of the authors if I've mangled your work in my implementation.

If that's the case, maybe you can [contribute a fix](https://github.com/justindujardin/mathy/){target=\_blank}?

### [A3C](https://arxiv.org/pdf/1602.01783){target=\_blank}

The work done by **Mnih et al.** in "Asynchronous Methods for Deep Reinforcement Learning" describes a CPU-friendly RL algorithm that can be trained for some tasks very rapidly. The `a3c` agent is based on this [tensorflow 2.0 implementation of A3C](https://medium.com/tensorflow/deep-reinforcement-learning-playing-cartpole-through-asynchronous-advantage-actor-critic-a3c-7eab2eea5296){target=\_blank}.

### [AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go){target=\_blank}

The work done by **[Silver et al.](https://arxiv.org/pdf/1712.01815.pdf){target=\_blank}** demonstrated the power of taking a principled approach to solving previously intractable problems. It inspired Mathy to use Neural Networks and reinforcement learning to pick combinations of rules to transform the tree with. This work lead directly to the removal of a layer of "composite" rules in Mathy for combining like terms and simplifying expressions.

### [Tidier Trees](https://reingold.co/tidier-drawings.pdf){target=\_blank}

The work done by **Reingold and Tilford** provides a relatively simple way to approach a difficult problem: how do you render arbitrarily large trees in a beautiful way? They provide an algorithm that does just that, enforcing a number of aesthetic rules on the trees it outputs. Mathy uses an implementation of this work to measure its trees for rendering.

### [UNREAL](https://deepmind.com/blog/article/reinforcement-learning-unsupervised-auxiliary-tasks){target=\_blank}

The work done by **[Jaderberg et al.](https://arxiv.org/pdf/1611.05397.pdf){target=\_blank}** explores the use of unsupervised auxiliary tasks to encourage RL agents to converge at a much quicker rate than they otherwise would. They find that using shared network weights to predict different things can help agents learned shared representations that both can benefit from. Mathy uses an auxiliary task for controlling the grouping of like terms in an expression, inspired by this work.

## Open Source

The scope of Mathy is large, and there are a few key contributions that have come from Open Source software that deserve special recognition.

### [FastAPI](https://fastapi.tiangolo.com/){target=\_blank}

FastAPI has a magnificent documentation site, and Mathy uses its template to structure its own site. A key insight from FastAPI that has made this site possible is the inclusion of code snippets from the website as tests that are run automatically. This makes maintaining tens or hundreds of code examples as easy as maintaining tests from a test suite.

??? note "MIT License"

    The MIT License (MIT)

    Copyright (c) 2018 Sebastián Ramírez

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

### [Fragile](https://github.com/FragileTech/fragile){target=\_blank}

Fragile implements a swarm planning aglorithm that Mathy uses to solve problems without a trained model. It is able to solve most of the built-in Mathy environments on a desktop computer without a GPU.

??? note "MIT License"

    Copyright 2020, Fragile Technologies

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/){target=\_blank}

The material theme for [MkDocs](https://www.mkdocs.org/){target=\_blank} makes creating beautiful documentation sites with mobile navigation and search as easy as writing some markdown files in a folder. Mathy uses this theme for its documentation site.

??? note "MIT License"

    The MIT License (MIT)

    Copyright © 2016 - 2019 Martin Donath

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

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

### [spaCy](https://github.com/explosion/spaCy/){target=\_blank}

spaCy has been an awesome inspiration for Mathy. It served to demonstrate that you can make Machine Learning as accessible as a pip install. Mathy uses a model packaging and loading system based on the one from spaCy.

??? note "MIT License"

    The MIT License (MIT)

    Copyright (C) 2016-2019 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.

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
