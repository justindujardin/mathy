Mathy was born out of a desire to have a free tool for working through algebra problems. Much research has informed the design of Mathy's computer algebra system, tree rendering, and learning environments.

## Research

### [Tidier Trees](https://reingold.co/tidier-drawings.pdf){target=\_blank}

The work done by **Reingold and Tilford** provides a relatively simple way to approach a complex problem: how do you beautifully render arbitrarily large trees? They provide an algorithm that does just that, enforcing several aesthetic rules on the trees it outputs. Mathy implements the algorithm to measure its trees for layout.

## Open Source

Mathy's scope is broad, and a few critical contributions from Open Source Software deserve special recognition.

### [Fragile](https://github.com/FragileTech/fragile){target=\_blank}

Fragile implements a swarm planning algorithm Mathy uses to solve problems without a trained model. It can solve most built-in Mathy environments on a desktop computer without a GPU.

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

When I began researching how to transform strings into expression trees, I came across a graphing calculator written in C#. The original parser and tokenizer came from Bob Brown at Microsoft, and Mathy still uses a modified version of this system. The clear and concise tokenizer/parser implementation has been endlessly helpful in porting Mathy to various languages.

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
