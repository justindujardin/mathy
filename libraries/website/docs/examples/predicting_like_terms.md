
# Predicting Like Terms [![Open Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justindujardin/mathy/blob/master/libraries/website/examples/predicting_like_terms.ipynb)

> This notebook is built using [thinc](https://thinc.ai){target=\_blank} and [Mathy](https://mathy.ai). 


Remember in Algebra how you had to combine "like terms" to simplify problems? 

You'd see expressions like `60 + 2x^3 - 6x + x^3 + 17x` that have **5** total terms but only **4** "like terms". 

That's because `2x^3` and `x^3` are like and `-6x` and `17x` are like, while `60` doesn't have any other terms that are like it.

Can we teach a model to predict that there are `4` like terms in the above expression?

Let's give it a shot using [Mathy](https://mathy.ai) to generate math problems and [thinc](https://thinc.ai) to build a regression model that outputs the number of like terms in each input problem.


```python
!pip install "thinc>=8.0.0a0" mathy
```

### Sketch a Model

Before we get started it can be good to have an idea of what input/output shapes we want for our model.

We'll convert text math problems into lists of lists of integers, so our example (X) type can be represented using thinc's `Ints2d` type.

The model will predict how many like terms there are in each sequence, so our output (Y) type can represented with the `Floats2d` type.

Knowing the thinc types we want enables us to create an alias for our model, so we only have to type out the verbose generic signature once.


```python
from typing import List
from thinc.api import Model
from thinc.types import Ints2d, Floats1d

ModelX = Ints2d
ModelY = Floats1d
ModelT = Model[List[ModelX], ModelY]
```

### Encode Text Inputs

Mathy generates ascii-math problems and we have to encode them into integers that the model can process. 

To do this we'll build a vocabulary of all the possible characters we'll see, and map each input character to its index in the list.

For math problems our vocabulary will include all the characters of the alphabet, numbers 0-9, and special characters like `*`, `-`, `.`, etc.


```python
from typing import List
from thinc.api import Model
from thinc.types import Ints2d, Floats1d
from thinc.api import Ops, get_current_ops

vocab = " .+-/^*()[]-01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def encode_input(text: str) -> ModelX:
    ops: Ops = get_current_ops()
    indices: List[List[int]] = []
    for c in text:
        if c not in vocab:
            raise ValueError(f"'{c}' missing from vocabulary in text: {text}")
        indices.append([vocab.index(c)])
    return ops.asarray2i(indices)
```

**Try It**

Let's try it out on some fixed data to be sure it works. 


```python
outputs = encode_input("4+2")
assert outputs[0][0] == vocab.index("4")
assert outputs[1][0] == vocab.index("+")
assert outputs[2][0] == vocab.index("2")
print(outputs)
```

    [[16]
     [ 2]
     [14]]


### Generate Math Problems

We'll use Mathy to generate random polynomial problems with a variable number of like terms. The generated problems will act as training data for our model.


```python
from typing import List, Optional, Set
import random
from mathy.problems import gen_simplify_multiple_terms

def generate_problems(number: int, exclude: Optional[Set[str]] = None) -> List[str]:
    if exclude is None:
        exclude = set()
    problems: List[str] = []
    while len(problems) < number:
        text, complexity = gen_simplify_multiple_terms(
            random.randint(2, 6),
            noise_probability=1.0,
            noise_terms=random.randint(2, 10),
            op=["+", "-"],
        )
        assert text not in exclude, "duplicate problem generated!"
        exclude.add(text)
        problems.append(text)
    return problems
```

**Try It**


```python
generate_problems(10)
```




    ['-7743l^3 + 3130r + -5826.8u - 4394r + 3g^4 - 1y - 1485u + 11w',
     '4d - -1525.5m^3 + 1w + 12l^4 + 3069.9w + -3559s - 1.8r + 6737.3l^4 - -2119l^4 - 3w + 1128.9a + -5600v - -2315b + 8.1u - -6832z',
     '-4868y^2 - 4548k + 9.6m + 3s - -7128d^4 + 6j^4 - 12v - 8.1t^4 + 1o^3 + 4c^4 - 2579o^3 - -4237.7q',
     '-4553l^2 - 11.7j + 10j - 8.3g - -5184m',
     '4o^2 + 2886u^3 + 5813q - 1u^3 + 4s - -6991u^3 + -9560a - -4774f + -1479z - 8.0f + 7x + 6.5h + -4397.2y + 12b',
     '1247m^4 + 3833q^2 + 1n - 11.7s - 1.3p - 618y^2 + -3821n + 2a - 2.4a - 11r - 4764w^3 + 4.5n - 2.2t + 572.9a - 3c^3',
     '1214.7f^4 + 11s - 2151k^4 - -7732q - 9q - 4l + -3697.3h + 3z + 5l - 7813.0p^3',
     '4m + 6x - 4u + 1f - 11m - 11.5d + 4.0z - 2n + 4386c^4 + 2.1q',
     '5h + 5.0h - 10.9a - 1517h + 2940o - -4178k^2 + -1748k^2',
     '1.5l - 8.1d - 7.4m^2 - 0.9a - -4580w - -8290.8k + 8.3j + 8g + -5722d - 5455s^2 + -5355r^2 + 11u^4']



### Count Like Terms

Now that we can generate input problems, we'll need a function that can count the like terms in each one and return the value for use as a label.

To accomplish this we'll use a few helpers from mathy to enumerate the terms and compare them to see if they're like.


```python
from typing import Optional, List, Dict
from mathy import MathExpression, ExpressionParser, get_terms, get_term_ex, TermEx
from mathy.problems import mathy_term_string

parser = ExpressionParser()

def count_like_terms(input_problem: str) -> int:
    expression: MathExpression = parser.parse(input_problem)
    term_nodes: List[MathExpression] = get_terms(expression)
    node_groups: Dict[str, List[MathExpression]] = {}
    for term_node in term_nodes:
        ex: Optional[TermEx] = get_term_ex(term_node)
        assert ex is not None, f"invalid expression {term_node}"
        key = mathy_term_string(variable=ex.variable, exponent=ex.exponent)
        if key == "":
            key = "const"
        if key not in node_groups:
            node_groups[key] = [term_node]
        else:
            node_groups[key].append(term_node)
    like_terms = 0
    for k, v in node_groups.items():
        if len(v) <= 1:
            continue
        like_terms += len(v)
    return like_terms
```

**Try It**


```python
assert count_like_terms("4x - 2y + q") == 0
assert count_like_terms("x + x + z") == 2
assert count_like_terms("4x + 2x - x + 7") == 3
```

### Generate Problem/Answer pairs

Now that we can generate problems, count the number of like terms in them, and encode their text into integers, we have the pieces required to generate random problems and answers that we can train a neural network with.

Let's write a function that will return a tuple of: the problem text, its encoded example form, and the output label.


```python
from typing import Tuple
from thinc.api import Ops, get_current_ops

def to_example(input_problem: str) -> Tuple[str, ModelX, ModelY]:
    ops: Ops = get_current_ops()
    encoded_input = encode_input(input_problem)
    like_terms = count_like_terms(input_problem)
    return input_problem, encoded_input, ops.asarray1f([like_terms])
```

**Try It**


```python
text, X, Y = to_example("x+2x")
assert text == "x+2x"
assert X[0] == vocab.index("x")
assert Y[0] == 2
print(text, X, Y)
```

    x+2x [[46]
     [ 2]
     [14]
     [46]] [2.]


### Build a Model

Now that we can generate X/Y values, let's define our model and verify that it can process a single input/output.

For this we'll use Thinc and the `define_operators` context manager to connect the pieces together using overloaded operators for `chain` and `clone` operations.


```python
from typing import List
from thinc.model import Model
from thinc.api import concatenate, chain, clone, list2ragged
from thinc.api import reduce_sum, Mish, with_array, Embed, residual

def build_model(n_hidden: int, dropout: float = 0.1) -> ModelT:
    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):
        model = (
            # Iterate over each element in the batch
            with_array(
                # Embed the vocab indices
                Embed(n_hidden, len(vocab), column=0)
                # Activate each batch of embedding sequences separately first
                >> Mish(n_hidden, dropout=dropout)
            )
            # Convert to ragged so we can use the reduction layers
            >> list2ragged()
            # Sum the features for each batch input
            >> reduce_sum()
            # Process with a small resnet
            >> residual(Mish(n_hidden, normalize=True)) ** 4
            # Convert (batch_size, n_hidden) to (batch_size, 1)
            >> Mish(1)
        )
    return model
```

**Try It**

Let's pass an example through the model to make sure we have all the sizes right.


```python
text, X, Y = to_example("14x + 2y - 3x + 7x")
m = build_model(12)
m.initialize([X], m.ops.asarray(Y, dtype="f"))
mY = m.predict([X])
print(mY.shape)
assert mY.shape == (1, 1)
```

    (1, 1)


### Generate Training Datasets

Now that we can generate examples and we have a model that can process them, let's generate random unique training and evaluation datasets.

For this we'll write another helper function that can generate (n) training examples and respects an exclude list to avoid letting examples from the training/test sets overlap.


```python
from typing import Tuple, Optional, Set, List

DatasetTuple = Tuple[List[str], List[ModelX], List[ModelY]]

def generate_dataset(
    size: int,
    exclude: Optional[Set[str]] = None,
) -> DatasetTuple:
    ops: Ops = get_current_ops()
    texts: List[str] = generate_problems(size, exclude=exclude)
    examples: List[ModelX] = []
    labels: List[ModelY] = []
    for i, text in enumerate(texts):
        text, x, y = to_example(text)
        examples.append(x)
        labels.append(y)

    return texts, examples, labels
```

**Try It**

Generate a small dataset to be sure everything is working as expected


```python
texts, x, y = generate_dataset(10)
assert len(texts) == 10
assert len(x) == 10
assert len(y) == 10
```

### Evaluate Model Performance

We're almost ready to train our model, we just need to write a function that will check a given trained model against a given dataset and return a 0-1 score of how accurate it was.

We'll use this function to print the score as training progresses and print final test predictions at the end of training.


```python
from typing import List
from wasabi import msg

def evaluate_model(
    model: ModelT,
    *,
    print_problems: bool = False,
    texts: List[str],
    X: List[ModelX],
    Y: List[ModelY],
):
    Yeval = model.predict(X)
    correct_count = 0
    print_n = 12
    if print_problems:
        msg.divider(f"eval samples max({print_n})")
    for text, y_answer, y_guess in zip(texts, Y, Yeval):
        y_guess = round(float(y_guess))
        correct = y_guess == int(y_answer)
        print_fn = msg.fail
        if correct:
            correct_count += 1
            print_fn = msg.good
        if print_problems and print_n > 0:
            print_n -= 1
            print_fn(f"Answer[{int(y_answer[0])}] Guess[{y_guess}] Text: {text}")
    if print_problems:
        print(f"Model predicted {correct_count} out of {len(X)} correctly.")
    return correct_count / len(X)

```

**Try It**

Let's try it out with an untrained model and expect to see a really sad score.


```python
texts, X, Y = generate_dataset(128)
m = build_model(12)
m.initialize(X, m.ops.asarray(Y, dtype="f"))
# Assume the model should do so poorly as to round down to 0
assert round(evaluate_model(m, texts=texts, X=X, Y=Y)) == 0
```

### Train/Evaluate a Model

The final helper function we need is one to train and evaluate a model given two input datasets. 

This function does a few things:

 1. Create an Adam optimizer we can use for minimizing the model's prediction error.
 2. Loop over the given training dataset (epoch) number of times.
 3. For each epoch, make batches of (batch_size) examples. For each batch(X), predict the number of like terms (Yh) and subtract the known answers (Y) to get the prediction error. Update the model using the optimizer with the calculated error.
 5. After each epoch, check the model performance against the evaluation dataset.
 6. Save the model weights for the best score out of all the training epochs.
 7. After all training is done, restore the best model and print results from the evaluation set.


```python
from thinc.api import Adam
from wasabi import msg
import numpy

def train_and_evaluate(
    model: ModelT,
    train_tuple: DatasetTuple,
    eval_tuple: DatasetTuple,
    *,
    lr: float = 3e-3,
    batch_size: int = 64,
    epochs: int = 48,
) -> float:
    (train_texts, train_X, train_y) = train_tuple
    (eval_texts, eval_X, eval_y) = eval_tuple
    msg.divider("Train and Evaluate Model")
    msg.info(f"Batch size = {batch_size}\tEpochs = {epochs}\tLearning Rate = {lr}")

    optimizer = Adam(lr)
    best_score: float = 0.0
    best_model: Optional[bytes] = None
    for n in range(epochs):
        loss = 0.0
        batches = model.ops.multibatch(batch_size, train_X, train_y, shuffle=True)
        for X, Y in batches:
            Y = model.ops.asarray(Y, dtype="float32")
            Yh, backprop = model.begin_update(X)
            err = Yh - Y
            backprop(err)
            loss += (err ** 2).sum()
            model.finish_update(optimizer)
        score = evaluate_model(model, texts=eval_texts, X=eval_X, Y=eval_y)
        if score > best_score:
            best_model = model.to_bytes()
            best_score = score
        print(f"{n}\t{score:.2f}\t{loss:.2f}")

    if best_model is not None:
        model.from_bytes(best_model)
    print(f"Evaluating with best model")
    score = evaluate_model(
        model, texts=eval_texts, print_problems=True, X=eval_X, Y=eval_y
    )
    print(f"Final Score: {score}")
    return score

```

We'll generate the dataset first, so we can iterate on the model without having to spend time generating examples for each run. This also ensures we have the same dataset across different model runs, to make it easier to compare performance.


```python
train_size = 1024 * 8
test_size = 2048
seen_texts: Set[str] = set()
msg.text(f"Generating train dataset with {train_size} examples...")
train_dataset = generate_dataset(train_size, seen_texts)
msg.text(f"Train set created with {train_size} examples.")
msg.text(f"Generating eval dataset with {test_size} examples...")
eval_dataset = generate_dataset(test_size, seen_texts)
msg.text(f"Eval set created with {test_size} examples.")
init_x = train_dataset[1][:2]
init_y = train_dataset[2][:2]
```

    [2Knerating train dataset with 8192 examples...[38;5;2m✔ Train set created with 8192 examples.[0m
    [2Knerating eval dataset with 2048 examples...[38;5;2m✔ Eval set created with 2048 examples.[0m


Finally, we can build, train, and evaluate our model!


```python
model = build_model(64)
model.initialize(init_x, init_y)
train_and_evaluate(
    model, train_dataset, eval_dataset, lr=2e-3, batch_size=64, epochs=32
)
```

    [1m
    ========================== Train and Evaluate Model ==========================[0m
    [38;5;4mℹ Batch size = 64      Epochs = 32     Learning Rate = 0.002[0m
    0	0.23	25283.75
    1	0.21	17535.96
    2	0.23	17329.38
    3	0.24	15842.63
    4	0.22	15396.75
    5	0.28	14760.94
    6	0.24	14000.68
    7	0.25	13252.41
    8	0.26	12263.78
    9	0.28	12130.36
    10	0.28	11368.95
    11	0.29	10993.20
    12	0.28	10709.44
    13	0.30	10305.06
    14	0.33	10134.89
    15	0.34	9738.52
    16	0.33	9579.92
    17	0.32	9091.71
    18	0.34	8950.21
    19	0.34	8553.15
    20	0.34	8320.76
    21	0.34	7905.48
    22	0.38	7880.38
    23	0.36	7484.32
    24	0.37	7348.58
    25	0.35	7158.16
    26	0.36	6754.80
    27	0.37	6588.11
    28	0.38	6534.72
    29	0.37	6266.50
    30	0.40	6176.08
    31	0.42	5852.44
    Evaluating with best model
    [1m
    ============================ eval samples max(12) ============================[0m
    [38;5;2m✔ Answer[6] Guess[6] Text: -7268s + 9c^4 - -3346u + -4891m + 12q^4 +
    3.8a + 8h + 10x - 1n^3 - 2.2k + 10b - -8598k - 5499b + 8496k + -5230b - 2r[0m
    [38;5;1m✘ Answer[2] Guess[3] Text: 11.0t - 11t + 2202f + -581a^3 - 10u[0m
    [38;5;2m✔ Answer[2] Guess[2] Text: 4085q^3 - 6667m - 9c + 2c + 3y[0m
    [38;5;2m✔ Answer[2] Guess[2] Text: 10h^3 + 11.0f^2 + 10r - 8091t^4 + 11b -
    1114x + 1r[0m
    [38;5;1m✘ Answer[2] Guess[4] Text: 0.9k + 575t^3 + 5975x - 4147l - 4.6j + 7r +
    0.6h^3 + 10b^2 + 1.8w - 4y + 7584.2w + 1q[0m
    [38;5;1m✘ Answer[2] Guess[4] Text: 1t^4 - 11z + 10.2h - 12s + 7374m + 954s +
    9q[0m
    [38;5;1m✘ Answer[2] Guess[4] Text: 10r + 12l - 7.8q + 2.9g - 8.3f + 3868.1a +
    7870p + -182p + 8.9z[0m
    [38;5;1m✘ Answer[2] Guess[3] Text: 3653x^4 - -8734m + 6418d - 12h^3 - 1069o -
    12p - -5812b - 7b - 9.2t - 2.1q + 7a^3[0m
    [38;5;1m✘ Answer[2] Guess[4] Text: 8x - 1a^2 - 6.4o + 1.2s^3 - -266y + 12f -
    -5511p + -3956n + 10.3v^4 + 0.5j + 3.5q - 3m - 8o[0m
    [38;5;1m✘ Answer[2] Guess[4] Text: 9335k + 6c^3 + 9.8p - 3.6u^4 - 3503f - 2.1h
    - -6713g^2 - 6.7d - -1433g^2 - 1j^4 - 10.9a^2 - 4.2s^2 + 4336n^3[0m
    [38;5;2m✔ Answer[6] Guess[6] Text: 0.9v^3 - 3x + 3g + 2k - 2926w^3 + 6b +
    -5993u^3 - 10.3t^2 + 0.5s^2 + 12z - 3585d - 1239z + 0.1d + 7980z - 761d -
    -1721f[0m
    [38;5;2m✔ Answer[2] Guess[2] Text: -713o - -6348.5x^4 - -4531z + 10.9j - 4o +
    -4799q - 11u^4 + -8290a - 9.5v[0m
    Model predicted 852 out of 2048 correctly.
    Final Score: 0.416015625





    0.416015625




```python

```
