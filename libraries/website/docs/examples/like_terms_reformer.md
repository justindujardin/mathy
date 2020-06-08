
# Like Terms Reformer [![Open Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/justindujardin/mathy/blob/master/libraries/website/examples/like_terms_reformer.ipynb)


> This notebook is built using [thinc](https://thinc.ai){target=_blank}, [PyTorch](https://pytorch.org/){target=_blank}, and the excellent [reformer_pytorch](https://github.com/lucidrains/reformer-pytorch){target=_blank} library. 

Remember in Algebra how you had to combine "like terms" to simplify problems?

You'd see expressions like `60 + 2x^3 - 6x + x^3 + 17x` that have **5** total terms but only **4** "like terms".

That's because `2x^3` and `x^3` are like and `-6x` and `17x` are like, while `60` doesn't have any other terms that are like it.

Can we teach a transformer to predict that there are `4` like terms in the above expression? Can we then show that the transformer has learned about like terms by inspecting its attention heads and layers?

Let's give it a shot using [reformer_pytorch](https://github.com/lucidrains/reformer-pytorch) to make our predictions.


```python
!pip install torch reformer_pytorch matplotlib wasabi pydantic typer
!pip install "git+git://github.com/justindujardin/thinc.git@feature/shim_thinc_optimizer#egg=thinc" --upgrade
try:
  !nvidia-smi
except Exception:
  pass
```


```python
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, cast, Optional

import numpy as np
import srsly
import torch
import typer
from pydantic import BaseModel
from reformer_pytorch import Recorder, ReformerLM
from thinc.shims.pytorch import PyTorchShim
from thinc.api import (
    to_numpy,
    Adam,
    Ops,
    PyTorchShim,
    PyTorchWrapper,
    fix_random_seed,
    get_current_ops,
    to_categorical,
    CategoricalCrossentropy,
    xp2torch,
)
from thinc.loss import Loss
from thinc.types import Floats1d, Floats2d, Ints1d, Ints2d

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from thinc.api import prefer_gpu, use_pytorch_for_gpu_memory
if prefer_gpu():
    use_pytorch_for_gpu_memory()
fix_random_seed(0)

TensorType = Ints2d
class PyTorchCrossEntropy(Loss):
    def __init__(
        self,
        *,
        normalize: bool = True,
        size_average: Optional[bool] = None,
        ignore_index: int = 0,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
    ):
        self.normalize = normalize
        self.reduction = reduction
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce

    def __call__(
        self, guesses: Floats2d, truths: Union[Ints1d, Floats2d]
    ) -> Tuple[Floats2d, float]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_torch_loss(
        self, guesses: "torch.Tensor", truths: "torch.Tensor", is_train: bool = False,
    ) -> "torch.Tensor":
        from torch.nn.functional import cross_entropy as torch_entropy

        if is_train:
            guesses.retain_grad()
        loss = torch_entropy(
            guesses,
            truths,
            size_average=self.size_average,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            reduce=self.reduce,
        )
        return loss.cpu()

    def get_grad(self, guesses: Floats2d, truths: Union[Ints1d, Floats2d]) -> Floats2d:
        import torch

        batch_tensor: torch.Tensor = xp2torch(guesses, requires_grad=True)
        need_transpose = len(batch_tensor.shape) == 3
        if need_transpose:
            batch_tensor = batch_tensor.transpose(2, 1)
        batch_labels = xp2torch(truths).long()
        batch_tensor.retain_grad()
        torch_loss = self.get_torch_loss(batch_tensor, batch_labels)
        torch_loss.backward()
        assert batch_tensor.grad is not None
        difference = batch_tensor.grad.data
        if need_transpose:
            difference = difference.transpose(2, 1)
        if self.normalize:
            difference = difference / batch_tensor.shape[0]
        return to_numpy(difference.cpu().numpy())

    def get_loss(
        self, guesses: Floats2d, truths: Union[Ints1d, Floats2d], is_train: bool = False
    ) -> float:
        batch_tensor = xp2torch(guesses, requires_grad=is_train)
        batch_labels = xp2torch(truths).long()
        if len(batch_tensor.shape) == 3:
            batch_tensor = batch_tensor.transpose(2, 1)
        loss = self.get_torch_loss(batch_tensor, batch_labels, is_train)
        return float(loss.cpu().numpy())


class ReformerLMConfig(BaseModel):
    num_tokens: int
    max_seq_len: int = 128
    dim: int = 512
    depth: int = 2
    bucket_size: int = 64
    heads: int = 4
    n_hashes: int = 4
    ff_chunks: int = 0
    lsh_dropout: float = 0.1


class MathyReformerConfig(BaseModel):
    folder: str
    train_file: str = "/dev/null"
    eval_file: str = "/dev/null"
    eval_batch_size: int = 128
    save_every: int = 100
    histogram_every: int = 100
    validate_every: int = 100
    print_every: int = 100
    use_cuda: bool = True
    use_profiler: bool = False

    batch_size: int = 512
    accumulate_every: int = 4
    learning_rate: float = 3e-4

    reformer: ReformerLMConfig

    @property
    def model_name(self) -> str:
        """Return the path to model.thinc file for this configuration"""
        return os.path.join(self.folder, "model.thinc")

    @property
    def log_dir(self) -> str:
        """Return the path to store tensorboard logs in"""
        return os.path.join(self.folder, "tensorboard")


DatasetTuple = Tuple[List[TensorType], List[TensorType]]


class MathyVocab:
    vocab: List[str]
    pad_token: int
    char_to_int: Dict[str, int]

    def __init__(self):
        self.vocab = [""] + list(
            " \t\n=.+-/^*()[]-?01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        self.pad_token = self.vocab.index("")
        self.char_to_int = {char: index for index, char in enumerate(self.vocab)}
        self.vocab_len = len(self.vocab)

    def decode_text(self, tokens: TensorType) -> str:
        """Decode an integer tensor to produce a string"""
        output: List[str] = []
        for token in tokens.tolist():
            token_index = int(token)
            assert token_index < self.vocab_len, f"invalid token: {token_index}"
            output.append(self.vocab[token_index])
        return "".join(output)

    def encode_text(self, text: str, pad_length: int = None) -> List[int]:
        """Encode text into a list of indices in the vocabulary"""
        if pad_length is not None:
            padding = [self.pad_token] * (pad_length - len(text))
        else:
            padding = []
        indices = [self.char_to_int[c] for c in text]
        return indices + padding


class MathyReformer:
    epoch: int
    config: MathyReformerConfig
    vocab: MathyVocab
    optimizer: Adam
    loss_fn: PyTorchCrossEntropy

    def save(self) -> None:
        model = os.path.join(self.config.folder, "model.thinc")
        self.net.to_disk(model)

    def __init__(
        self,
        config: MathyReformerConfig,
        vocab: MathyVocab,
        must_exist: bool = False,
        record_attention: bool = False,
    ):
        self.loss_fn = PyTorchCrossEntropy(normalize=False, reduction="sum")
        model = os.path.join(config.folder, "model.thinc")
        model_config = os.path.join(config.folder, "config.json")
        if os.path.exists(model):
            config = MathyReformerConfig(**srsly.read_json(model_config))
        elif must_exist:
            raise ValueError(f"model not found: {model}")
        else:
            Path(model_config).parent.mkdir(exist_ok=True, parents=True)
            srsly.write_json(model_config, config.dict())
            print(f"wrote model config: {model_config}")
        self.config = config
        self.vocab = vocab
        self.reformer = ReformerLM(**config.reformer.dict())
        self.net = PyTorchWrapper(self.reformer)
        self.epoch = 0
        if os.path.exists(model):
            print(f"loading model: {model}")
            self.net.from_disk(model)
            # self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # self.epoch = checkpoint.get("epoch", 0)
        if record_attention:
            self.reformer = Recorder(self.reformer)
        self.optimizer = Adam(config.learning_rate)
        print(f"model epoch: {self.epoch}")


def main(
    folder: str = "training/reformer/dev_reformer",
    train_file="like_terms_prediction.train.txt",
    eval_file="like_terms_prediction.eval.txt",
):
    vocab = MathyVocab()
    config = MathyReformerConfig(
        folder=folder,
        train_file=train_file,
        eval_file=eval_file,
        reformer=ReformerLMConfig(num_tokens=vocab.vocab_len),
    )
    print(f"Folder: {config.folder}")
    print(f"Config: {json.dumps(config.dict(), indent=2)}")
    reformer: MathyReformer = MathyReformer(
        config=config, vocab=vocab,
    )
    pr = None
    if config.use_profiler:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        print(f"PROFILER: recording")
    train(reformer, config=config, epochs=3000)

    if config.use_profiler:
        assert pr is not None
        profile_path = os.path.join(config.folder, "training.profile")
        pr.disable()
        pr.dump_stats(profile_path)
        print(f"PROFILER: saved {profile_path}")


def load_dataset(file_name: str, pad_length: int, model: MathyReformer) -> DatasetTuple:
    """Load a dataset where question/answer pairs are separated by newlines, and
    pad the outputs to match the transformer sequence length."""
    with open(file_name) as f:
        lines = f.readlines()
    in_lines: List[TensorType] = []
    out_lines: List[TensorType] = []
    for i, l in tqdm(enumerate(lines), desc=f"loading dataset: {file_name}"):
        encoded = np.asarray(model.vocab.encode_text(l, pad_length), dtype="int64")
        if i % 2 == 0:
            in_lines.append(encoded)
        else:
            out_lines.append(encoded)
    assert len(in_lines) == len(out_lines), "in/out files must have 1-to-1 line mapping"
    in_lines = model.net.ops.asarray2i(in_lines, dtype="int64")
    out_lines = model.net.ops.asarray2i(out_lines, dtype="int64")
    return in_lines, out_lines


def cycle(loader):
    """Cycle through a dataset forever"""
    while True:
        for data in loader:
            yield data


def evaluate_model(
    model: MathyReformer, dataset: DatasetTuple
) -> Tuple[float, float, List[str]]:
    """Evaluate a model on a dataset and return a tuple of the total number
    of problems evaluated, the number answered correctly, and the total loss """
    with torch.no_grad():
        ops: Ops = get_current_ops()
        batches = ops.multibatch(
            model.config.eval_batch_size, dataset[0], dataset[1], shuffle=True
        )
        model.reformer.eval()
        losses: List[float] = []
        correct: int = 0
        total: int = len(dataset[0])
        print_max = 3
        printed = 0
        texts = []
        for batch_with_labels in batches:
            batch, batch_labels = batch_with_labels
            # Check correct/incorrect answers
            # TODO: remove the need for this torch/ops/long conversion
            batch = xp2torch(ops.asarray(batch, dtype="int64")).long()
            if model.config.use_cuda:
                batch = batch.cuda()
            prediction = model.net(batch, is_train=False)[0]
            answer: Any
            for X, label, answer in zip(batch, batch_labels, prediction):
                label = xp2torch(model.net.ops.asarray2i(label))
                expected = model.vocab.decode_text(label).replace("\n", "")
                # argmax resolves the class probs to ints, and squeeze removes extra dim
                answer = model.vocab.decode_text(answer.argmax(-1).squeeze())
                if "\n" in answer:
                    answer = answer[0 : answer.index("\n")]
                if printed < print_max:
                    printed += 1
                    question = model.vocab.decode_text(X).replace("\n", "")
                    outcome = "WRONG" if expected != answer else "RIGHT"
                    print_text = f"{outcome} | answer: {expected} | model: {answer} | question: {question}"
                    texts.append(print_text)
                if answer == expected:
                    correct += 1
            batch_loss = get_batch_loss(model, batch_with_labels, prediction=prediction)
            batch_loss.squeeze()
            losses.append(float(batch_loss.mean()))
        ratio = correct / total
        print(
            f"evaluation accuracy: {int(ratio * 100)}% | correct: ({correct}/{total})"
        )
        loss = model.net.ops.asarray1f(losses).mean()
        return loss, ratio, texts


def train(
    model: MathyReformer, *, config: MathyReformerConfig, epochs: int,
):
    summary = SummaryWriter(os.path.join(config.log_dir), flush_secs=30)
    data_train = load_dataset(config.train_file, config.reformer.max_seq_len, model)
    data_val = load_dataset(config.eval_file, config.reformer.max_seq_len, model)
    ops: Ops = get_current_ops()
    train_loader = cycle(
        model.net.ops.multibatch(
            model.config.batch_size, data_train[0], data_train[1], shuffle=True
        )
    )
    try:
        for i in range(epochs):
            model.epoch += 1
            model.reformer.train()

            step_start = time.time()
            losses: List[float] = []
            for __ in range(config.accumulate_every):
                batch_loss = get_batch_loss(model, next(train_loader))
                batch_loss.squeeze()
                losses.append(float(batch_loss.mean()))

            loss = float(
                model.net.ops.asarray1f(losses).mean() / config.accumulate_every
            )
            print(".", end="", flush=True)
            step_end = time.time()
            summary.add_scalar("metrics/epoch_time", step_end - step_start, model.epoch)
            summary.add_scalar("loss/train", float(loss), model.epoch)
            torch.nn.utils.clip_grad_norm_(model.reformer.parameters(), 0.5)
            # model.optimizer.step()
            # run before zero-grad
            if i % config.histogram_every == 0:
                for tag, value in model.reformer.named_parameters():
                    tag = tag.replace(".", "/")
                    summary.add_histogram(
                        tag + "/data", value.data.cpu().numpy(), model.epoch
                    )
                    summary.add_histogram(
                        tag + "/gradient", value.grad.data.cpu().numpy(), model.epoch
                    )
            model.net.finish_update(model.optimizer)
            model.optimizer.step_schedules()
            # model.optimizer.zero_grad()

            if i % config.print_every == 0:
                print(f"step: {model.epoch} | loss: {loss}")

            if i % config.save_every == 0:
                if i > 0:
                    model.save()
            if i % config.validate_every == 0:
                eval_loss, eval_win_pct, texts = evaluate_model(model, data_val)
                print(f"loss_train: {loss} | loss_eval: {eval_loss}")
                summary.add_scalar("loss/eval", float(eval_loss), model.epoch)
                summary.add_scalar(
                    "metrics/eval_correct_pct", eval_win_pct, model.epoch
                )
                for i, text in enumerate(texts):
                    summary.add_text(f"metrics/eval_sample_{i}", text, model.epoch)
    except KeyboardInterrupt:
        pass

    summary.close()
    model.save()


def get_batch_loss(
    model: MathyReformer,
    data: Tuple[TensorType, Tuple[TensorType, TensorType]],
    prediction: TensorType = None,
) -> Floats1d:
    x, label = data

    # Allow passing a prediction to avoid duplicate model calls
    backprop = None
    if prediction is None:
        prediction, backprop = model.net.begin_update(xp2torch(x.astype("int64")))

    # TODO: remove this reshape when/if to_categorical does it automatically
    # label = to_categorical(label, n_classes=prediction.shape[-1]).reshape(
    #     prediction.shape
    # )
    loss = model.loss_fn.get_loss(prediction, label)
    if backprop is not None:
        d_loss = xp2torch(model.loss_fn.get_grad(prediction, label))
        if model.config.use_cuda:
            d_loss = d_loss.cuda()
        backprop(d_loss)
    return model.net.ops.asarray1f(loss)

```


```python
main()
```

    Folder: training/reformer/dev_reformer
    Config: {
      "folder": "training/reformer/dev_reformer",
      "train_file": "like_terms_prediction.train.txt",
      "eval_file": "like_terms_prediction.eval.txt",
      "eval_batch_size": 128,
      "save_every": 100,
      "histogram_every": 100,
      "validate_every": 100,
      "print_every": 100,
      "use_cuda": true,
      "use_profiler": false,
      "batch_size": 512,
      "accumulate_every": 4,
      "learning_rate": 0.0003,
      "reformer": {
        "num_tokens": 80,
        "max_seq_len": 128,
        "dim": 512,
        "depth": 2,
        "bucket_size": 64,
        "heads": 4,
        "n_hashes": 4,
        "ff_chunks": 0,
        "lsh_dropout": 0.1
      }
    }
    wrote model config: training/reformer/dev_reformer/config.json


    loading dataset: like_terms_prediction.train.txt: 6670it [00:00, 66698.16it/s]

    model epoch: 0


    loading dataset: like_terms_prediction.train.txt: 400000it [00:06, 66001.07it/s]
    loading dataset: like_terms_prediction.eval.txt: 2000it [00:00, 48498.31it/s]


    .step: 1 | loss: 1219.503662109375
    evaluation accuracy: 0% | correct: (5/1000)
    loss_train: 1219.503662109375 | loss_eval: 797.3507
    ....................................................................................................step: 101 | loss: 163.42306518554688
    evaluation accuracy: 48% | correct: (483/1000)
    loss_train: 163.42306518554688 | loss_eval: 155.58597
    ....................................................................................................step: 201 | loss: 149.3126678466797
    evaluation accuracy: 51% | correct: (519/1000)
    loss_train: 149.3126678466797 | loss_eval: 144.91866
    ....................................................................................................step: 301 | loss: 138.98428344726562
    evaluation accuracy: 57% | correct: (571/1000)
    loss_train: 138.98428344726562 | loss_eval: 133.37936
    ....................................................................................................step: 401 | loss: 135.4252471923828
    evaluation accuracy: 56% | correct: (566/1000)
    loss_train: 135.4252471923828 | loss_eval: 132.99727
    ....................................................................................................step: 501 | loss: 133.71649169921875
    evaluation accuracy: 57% | correct: (574/1000)
    loss_train: 133.71649169921875 | loss_eval: 132.64777
    ....................................................................................................step: 601 | loss: 129.60745239257812
    evaluation accuracy: 61% | correct: (613/1000)
    loss_train: 129.60745239257812 | loss_eval: 126.70494
    ....................................................................................................step: 701 | loss: 122.04228210449219
    evaluation accuracy: 62% | correct: (626/1000)
    loss_train: 122.04228210449219 | loss_eval: 119.77521
    ....................................................................................................step: 801 | loss: 114.01881408691406
    evaluation accuracy: 64% | correct: (648/1000)
    loss_train: 114.01881408691406 | loss_eval: 113.77133
    ....................................................................................................step: 901 | loss: 108.57820129394531
    evaluation accuracy: 65% | correct: (655/1000)
    loss_train: 108.57820129394531 | loss_eval: 110.04442
    ....................................................................................................step: 1001 | loss: 108.431640625
    evaluation accuracy: 65% | correct: (652/1000)
    loss_train: 108.431640625 | loss_eval: 109.68504
    ....................................................................................................step: 1101 | loss: 114.97943115234375
    evaluation accuracy: 64% | correct: (648/1000)
    loss_train: 114.97943115234375 | loss_eval: 111.26749
    ....................................................................................................step: 1201 | loss: 107.43222045898438
    evaluation accuracy: 65% | correct: (656/1000)
    loss_train: 107.43222045898438 | loss_eval: 107.309746
    ....................................................................................................step: 1301 | loss: 108.95185852050781
    evaluation accuracy: 65% | correct: (654/1000)
    loss_train: 108.95185852050781 | loss_eval: 107.30882
    ....................................................................................................step: 1401 | loss: 112.26199340820312
    evaluation accuracy: 65% | correct: (651/1000)
    loss_train: 112.26199340820312 | loss_eval: 106.98419
    ....................................................................................................step: 1501 | loss: 106.8240737915039
    evaluation accuracy: 65% | correct: (653/1000)
    loss_train: 106.8240737915039 | loss_eval: 106.37789
    ....................................................................................................step: 1601 | loss: 105.31365966796875
    evaluation accuracy: 66% | correct: (665/1000)
    loss_train: 105.31365966796875 | loss_eval: 105.86846
    ....................................................................................................step: 1701 | loss: 108.36333465576172
    evaluation accuracy: 66% | correct: (661/1000)
    loss_train: 108.36333465576172 | loss_eval: 106.76781
    ....................................................................................................step: 1801 | loss: 104.20068359375
    evaluation accuracy: 65% | correct: (654/1000)
    loss_train: 104.20068359375 | loss_eval: 105.77577
    ....................................................................................................step: 1901 | loss: 106.36077880859375
    evaluation accuracy: 66% | correct: (661/1000)
    loss_train: 106.36077880859375 | loss_eval: 106.480286
    ....................................................................................................step: 2001 | loss: 105.81416320800781
    evaluation accuracy: 66% | correct: (660/1000)
    loss_train: 105.81416320800781 | loss_eval: 105.802826
    ....................................................................................................step: 2101 | loss: 107.98883056640625
    evaluation accuracy: 65% | correct: (652/1000)
    loss_train: 107.98883056640625 | loss_eval: 108.55791
    ....................................................................................................step: 2201 | loss: 108.25115966796875
    evaluation accuracy: 65% | correct: (656/1000)
    loss_train: 108.25115966796875 | loss_eval: 110.0211
    ....................................................................................................step: 2301 | loss: 103.73287963867188
    evaluation accuracy: 66% | correct: (661/1000)
    loss_train: 103.73287963867188 | loss_eval: 105.552765
    ....................................................................................................step: 2401 | loss: 94.80268096923828
    evaluation accuracy: 70% | correct: (704/1000)
    loss_train: 94.80268096923828 | loss_eval: 91.30623
    ....................................................................................................step: 2501 | loss: 63.13200378417969
    evaluation accuracy: 80% | correct: (809/1000)
    loss_train: 63.13200378417969 | loss_eval: 58.11401
    ....................................................................................................step: 2601 | loss: 22.000757217407227
    evaluation accuracy: 94% | correct: (941/1000)
    loss_train: 22.000757217407227 | loss_eval: 21.18526
    ....................................................................................................step: 2701 | loss: 9.245500564575195
    evaluation accuracy: 95% | correct: (950/1000)
    loss_train: 9.245500564575195 | loss_eval: 14.451437
    ....................................................................................................step: 2801 | loss: 9.164619445800781
    evaluation accuracy: 99% | correct: (991/1000)
    loss_train: 9.164619445800781 | loss_eval: 3.9043887
    ....................................................................................................step: 2901 | loss: 5.663301944732666
    evaluation accuracy: 98% | correct: (989/1000)
    loss_train: 5.663301944732666 | loss_eval: 4.055765
    ...................................................................................................
