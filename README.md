## Mathy

A reinforcement learning agent that learns to solve math problems with step-by-step explanations through self-practice on generated problems from user specified template functions.


```bash
COMBINE_LIKE_TERMS_1 - SIX_TERMS...
-- cs -- -- ag -- | 36 | -01 | initial-state             | 13z^2 + 2x + 16z + 21x + 13z + 20z^2
-- cs -- -- ag -- | 36 | 009 | commutative swap          | 16z + (13z^2 + 2x) + 21x + 13z + 20z^2
-- cs -- -- ag -- | 35 | 017 | commutative swap          | 13z + (16z + (13z^2 + 2x) + 21x) + 20z^2
-- cs -- df ag -- | 34 | 007 | associative group         | 13z + (16z + (13z^2 + 2x + 21x)) + 20z^2
ca cs dm -- ag -- | 33 | 017 | distributive factoring    | 13z + (16z + (13z^2 + (2 + 21) * x)) + 20z^2
-- cs -- -- ag -- | 32 | 015 | constant arithmetic       | 13z + (16z + (13z^2 + 23x)) + 20z^2
-- cs -- -- ag -- | 31 | 013 | commutative swap          | 13z + (16z + (23x + 13z^2)) + 20z^2
-- cs -- -- ag -- | 30 | 003 | associative group         | 13z + (16z + (23x + 13z^2) + 20z^2)
-- cs -- df ag -- | 29 | 007 | associative group         | 13z + (16z + (23x + 13z^2 + 20z^2))
-- cs -- df ag -- | 28 | 007 | associative group         | 13z + 16z + (23x + 13z^2 + 20z^2)
ca cs dm df ag -- | 27 | 017 | distributive factoring    | 13z + 16z + (23x + (13 + 20) * z^2)
ca cs dm -- ag -- | 26 | 003 | distributive factoring    | (13 + 16) * z + (23x + (13 + 20) * z^2)
ca cs dm -- ag -- | 25 | 001 | constant arithmetic       | 29z + (23x + (13 + 20) * z^2)
-- cs -- -- ag -- | 24 | 009 | constant arithmetic       | 29z + (23x + 33z^2)
```

### Development Setup

You need a Python3 (probably virutal) environment with the correct set of dependencies for your CPU or GPU training environment. Here's how to configure one assuming that you have python and the `virtualenv` package installed.

#### CPU Environment

```bash
virtualenv -p python3.6 .env
source .env/bin/activate
pip install -r requirements.txt -r requirements.cpu.txt
```

#### GPU Environment

```bash
virtualenv -p python3.6 .env
source .env/bin/activate
pip install -r requirements.txt -r requirements.gpu.txt
```

### Test suite

Mathy has a suite of tests to ensure the math and agent libraries
work as expected. Run them with `pytest` once you've setup and activated you virtualenv

```bash
python -m pytest --cov=mathy mathy
```

### Agent Training

Math is able to manipulate math expressions using a reinforcement learning agent that interacts with our constructed `MathEnvironment`. To train a new agent use the `main.py` script. It has the following arguments:

```bash
(.env) mathy > python main.py --help
usage: main.py [-h] [-l None] [-t] [-v] model_dir [transfer_from]

positional arguments:
  model_dir             The name of the model to train. This changes the
                        output folder.
  transfer_from         The name of another model to warm start this one from.
                        Think Transfer Learning

optional arguments:
  -h, --help            show this help message and exit
  -l None, --lesson-id None
                        The lesson plan to execute by ID
  -t, --initial-train   When true, train the network on everything in
                        `examples.json` in the checkpoint directory
  -v, --verbose         When true, print all problem moves rather than just
                        during evaluation
```

Let's check to ensure the agent is working by solving some two and three term problems using the brute-force power
of the Monte Carlo Tree Search to provide good results without a pretrained model. We'll launch the main training 
script with a verbose argument to see the moves and using the `quick` lesson plan so it doesn't take a long time to
run through a few problem types.

```bash
(.env) mathy > python main.py ./trained/new_model -l dev -v
-- init math model in: ./trained/new_model/train
init model dir: None
[Lesson:0]
lesson order: ['two_terms', 'three_terms']

COMBINE_LIKE_TERMS_1 - TWO_TERMS...
-- cs -- df -- -- | 12 | -01 | initial-state             | 2x + 4x
ca cs dm -- -- -- | 12 | 003 | distributive factoring    | (2 + 4) * x
-- -- -- -- -- -- | 11 | 001 | constant arithmetic       | 6x
TWO_TERMS [2/1] -- duration(0:00:06.820133) outcome(solved)
[skip training] only have 46 observations, but need at least 1024 before training

COMBINE_LIKE_TERMS_1 - THREE_TERMS...
-- cs -- df ag -- | 18 | -01 | initial-state             | 19y + 20y + 17y
ca cs dm -- -- -- | 18 | 003 | distributive factoring    | (19 + 20) * y + 17y
-- cs -- df -- -- | 17 | 001 | constant arithmetic       | 39y + 17y
ca cs dm -- -- -- | 16 | 003 | distributive factoring    | (39 + 17) * y
-- -- -- -- -- -- | 15 | 001 | constant arithmetic       | 56y
THREE_TERMS [4/1] -- duration(0:00:01.821146) outcome(solved)
[skip training] only have 50 observations, but need at least 1024 before training
```


### Credits

The math parser and expression class hierarchies were originally based on the awesome [Silverlight Graphing Calculator](https://code.msdn.microsoft.com/silverlight/Silverlight-Graphing-fb30536e) project.

The MCTS and AlphaGo framework for the original Python version comes from [Alpha Zero General](https://github.com/suragnair/alpha-zero-general)
