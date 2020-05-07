import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, cast

import numpy as np
import srsly
import torch
import typer
from pydantic import BaseModel
from reformer_pytorch import Recorder, ReformerLM
from thinc.api import (
    Adam,
    Ops,
    PyTorchWrapper,
    fix_random_seed,
    get_current_ops,
    to_categorical,
    xp2torch,
)
from thinc.loss import Loss
from thinc.types import Floats1d, Floats2d, Ints1d, Ints2d
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

fix_random_seed(0)

TensorType = Ints2d


class CategoricalCrossentropy(Loss):
    def __init__(self, *, normalize: bool = True):
        self.normalize = normalize

    def __call__(
        self, guesses: Floats2d, truths: Union[Ints1d, Floats2d]
    ) -> Tuple[Floats2d, float]:
        return self.get_grad(guesses, truths), self.get_loss(guesses, truths)

    def get_grad(self, guesses: Floats2d, truths: Union[Ints1d, Floats2d]) -> Floats2d:
        if truths.ndim != guesses.ndim:
            # transform categorical values to one-hot encoding
            target = to_categorical(cast(Ints1d, truths), n_classes=guesses.shape[-1])
        else:  # pragma: no cover
            target = cast(Floats2d, truths)
        if guesses.shape != target.shape:  # pragma: no cover
            err = f"Cannot calculate CategoricalCrossentropy loss: mismatched shapes: {guesses.shape} vs {target.shape}."
            raise ValueError(err)
        # NOTE: these showed up taking a ton of time in colab profiles when using cupy.
        # .any() must not be cheap on the GPU.
        #
        # if guesses.any() > 1 or guesses.any() < 0:  # pragma: no cover
        #     err = f"Cannot calculate CategoricalCrossentropy loss with guesses outside the [0,1] interval."
        #     raise ValueError(err)
        # if target.any() > 1 or target.any() < 0:  # pragma: no cover
        #     err = f"Cannot calculate CategoricalCrossentropy loss with truth values outside the [0,1] interval."
        #     raise ValueError(err)
        difference = guesses - target
        if self.normalize:
            difference = difference / guesses.shape[0]
        return difference

    def get_loss(self, guesses: Floats2d, truths: Union[Ints1d, Floats2d]) -> float:
        d_truth = self.get_grad(guesses, truths)
        # TODO: Add overload for axis=None case to sum
        return (d_truth ** 2).sum()  # type: ignore


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
    loss_fn: CategoricalCrossentropy

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
        self.loss_fn = CategoricalCrossentropy(normalize=False)
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
        batch = xp2torch(ops.asarray(batch, dtype="int64"))
        prediction = model.net.predict(batch)
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
        losses.append(float(batch_loss))
    ratio = correct / total
    print(f"evaluation accuracy: {int(ratio * 100)}% | correct: ({correct}/{total})")
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
                losses.append(float(batch_loss))

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
    label = to_categorical(label, n_classes=prediction.shape[-1]).reshape(
        prediction.shape
    )
    d_loss, loss = model.loss_fn(prediction, label)
    if backprop is not None:
        backprop(torch.stack([xp2torch(l) for l in d_loss]))
    return model.net.ops.asarray1f(loss)


if __name__ == "__main__":
    import typer

    typer.run(main)
