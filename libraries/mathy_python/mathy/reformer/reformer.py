import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
from typing import List, Tuple, Union

from pydantic import BaseModel
from reformer_pytorch import Recorder, ReformerLM
import srsly
from thinc.api import Model, Ops, PyTorchWrapper, fix_random_seed, get_current_ops
from thinc.types import Ints2d
import torch
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm import tqdm
import typer

fix_random_seed(0)
# TODO: Replace with thinc to make swapping frameworks easier
TensorType = torch.Tensor


class ReformerLMConfig(BaseModel):
    # ReformerLM config
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
    use_cuda: bool = False

    batch_size: int = 512
    accumulate_every: int = 4
    learning_rate: float = 3e-4

    reformer: ReformerLMConfig

    @property
    def model_name(self) -> str:
        """Return the path to model.torch file for this configuration"""
        return os.path.join(self.folder, "model.torch")

    @property
    def log_dir(self) -> str:
        """Return the path to store tensorboard logs in"""
        return os.path.join(self.folder, "tensorboard")


DatasetTuple = Tuple[List[TensorType], List[Tuple[TensorType, int]]]


class MathyVocab:
    vocab: List[str]
    pad_token: int
    char_to_int: Dict[str, int]

    def __init__(self):
        self.vocab = ["", " ", "\t", "\n"] + list(
            "=.+-/^*()[]-?01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        self.pad_token = self.vocab.index("")
        self.char_to_int = {char: index for index, char in enumerate(self.vocab)}
        self.vocab_len = len(self.vocab)

    def decode_text(self, tokens: TensorType) -> str:
        """Decode a list of integer tensors to produce a string"""
        output: List[str] = []
        for token in tokens.tolist():
            token_index = int(token)
            assert token_index < self.vocab_len, "invalid token"
            output.append(self.vocab[token_index])
        return "".join(output)

    def encode_text(
        self, text: str, pad_length: int = None, include_batch: bool = False
    ) -> TensorType:
        """Encode text into a list of indices in the vocabulary"""
        ops: Ops = get_current_ops()
        if pad_length is not None:
            padding = [self.pad_token] * (pad_length - len(text))
        else:
            padding = []
        values = [self.char_to_int[c] for c in text] + padding
        if include_batch:
            values = [values]
        return ops.asarray2i(values, dtype="int64")


class MathyReformer:
    epoch: int
    config: MathyReformerConfig
    vocab: MathyVocab
    optimizer: torch.optim.Adam
    reformer: Union[ReformerLM, Recorder]

    def __init__(
        self,
        config: MathyReformerConfig,
        vocab: MathyVocab,
        must_exist: bool = False,
        record_attention: bool = False,
    ):
        model = os.path.join(config.folder, "model.torch")
        model_config = os.path.join(config.folder, "config.json")
        if os.path.exists(model):
            config = MathyReformerConfig(**srsly.read_json(model_config))
        elif must_exist:
            raise ValueError(f"model not found: {model}")
        else:
            Path(model_config).parent.mkdir(exist_ok=True, parents=True)
            srsly.write_json(model_config, config.dict())
            print(f"wrote model config: {model_config}")
        self.reformer = ReformerLM(**config.reformer.dict())
        if config.use_cuda:
            self.reformer.cuda()
        self.optimizer = torch.optim.Adam(
            self.reformer.parameters(), lr=config.learning_rate
        )
        self.config = config
        self.vocab = vocab
        self.epoch = 0
        if os.path.exists(model):
            print(f"loading model: {model}")
            dev = "cpu" if config.use_cuda is False else "cuda"
            checkpoint = torch.load(model, map_location=torch.device(dev))
            self.reformer.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint.get("epoch", 0)
        if record_attention:
            self.reformer = Recorder(self.reformer)
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
    train(reformer, config=config, epochs=3000)


def load_dataset(file_name: str, pad_length: int, vocab: MathyVocab) -> DatasetTuple:
    """Load a dataset where question/answer pairs are separated by newlines, and
    pad the outputs to match the transformer sequence length."""
    with open(file_name) as f:
        lines = f.readlines()
    in_lines: List[TensorType] = []
    out_lines: List[Tuple[TensorType, int]] = []
    for i, l in tqdm(enumerate(lines), desc=f"loading dataset: {file_name}"):
        encoded = vocab.encode_text(l, pad_length)
        if i % 2 == 0:
            in_lines.append(encoded)
        else:
            out_lines.append((encoded, len(l)))
    assert len(in_lines) == len(out_lines), "in/out files must have 1-to-1 line mapping"
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
    loss: float = 0.0
    correct: int = 0
    total: int = len(dataset[0])
    print_max = 3
    printed = 0
    texts = []
    with torch.no_grad():
        for batch_with_labels in batches:
            batch, batch_labels = batch_with_labels
            # Check correct/incorrect answers
            # TODO: remove the need for this torch/ops/long conversion
            batch = torch.from_numpy(ops.asarray(batch))
            prediction = model.reformer(batch)
            answer: Any
            for X, label_and_len, answer in zip(batch, batch_labels, prediction):
                label = label_and_len[0]
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
            loss += get_batch_loss(
                model, batch_with_labels, prediction=prediction
            ).item()
    ratio = correct / total
    print(f"evaluation accuracy: {int(ratio * 100)}% | correct: ({correct}/{total})")
    return loss, ratio, texts


def get_batch_loss(
    model: MathyReformer,
    data: Tuple[TensorType, Tuple[TensorType, TensorType]],
    prediction: TensorType = None,
):
    ops: Ops = get_current_ops()
    x, y = data
    x = torch.from_numpy(ops.asarray(x))
    label = torch.from_numpy(ops.asarray([l[0] for l in y]))
    label_len = torch.from_numpy(ops.asarray([l[1] for l in y]))

    # Allow passing a prediction to avoid duplicate model calls
    if prediction is None:
        prediction = model.reformer(x)
    # y is a tuple of (label, length) to allow unpadding
    label_max_len = int(label_len.max())
    label = label.narrow(1, 0, label_max_len)
    pred = prediction.narrow(1, 0, label_max_len)
    loss = cross_entropy(pred.transpose(1, 2), label, reduction="mean")
    return loss


def train(
    model: MathyReformer, *, config: MathyReformerConfig, epochs: int,
):
    summary = SummaryWriter(os.path.join(config.log_dir), flush_secs=30)
    data_train = load_dataset(
        config.train_file, config.reformer.max_seq_len, model.vocab
    )
    data_val = load_dataset(config.eval_file, config.reformer.max_seq_len, model.vocab)
    ops: Ops = get_current_ops()
    train_loader = cycle(
        ops.multibatch(
            model.config.batch_size, data_train[0], data_train[1], shuffle=True
        )
    )
    loss: TensorType = torch.zeros(1)

    def save():
        print(
            f"save: {config.model_name} | step: {model.epoch} | loss_train: {loss.item()}"
        )
        torch.save(
            {
                "epoch": model.epoch,
                "model_state_dict": model.reformer.state_dict(),
                "optimizer_state_dict": model.optimizer.state_dict(),
            },
            config.model_name,
        )

    try:
        for i in range(epochs):
            model.epoch += 1
            model.reformer.train()

            step_start = time.time()
            for __ in range(config.accumulate_every):
                loss = get_batch_loss(model, next(train_loader))
                loss.backward()
            print(".", end="", flush=True)
            step_end = time.time()
            summary.add_scalar("metrics/epoch_time", step_end - step_start, model.epoch)
            summary.add_scalar("loss/train", loss, model.epoch)
            torch.nn.utils.clip_grad_norm_(model.reformer.parameters(), 0.5)
            model.optimizer.step()
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
            model.optimizer.zero_grad()

            if i % config.print_every == 0:
                print(f"step: {model.epoch} | loss: {loss.item()}")

            if i % config.save_every == 0:
                if i > 0:
                    save()
            if i % config.validate_every == 0:
                eval_loss, eval_win_pct, texts = evaluate_model(model, data_val)
                print(f"loss_train: {loss.item()} | loss_eval: {eval_loss}")
                summary.add_scalar("loss/eval", eval_loss, model.epoch)
                summary.add_scalar(
                    "metrics/eval_correct_pct", eval_win_pct, model.epoch
                )
                for i, text in enumerate(texts):
                    summary.add_text(f"metrics/eval_sample_{i}", text, model.epoch)
    except KeyboardInterrupt:
        pass

    summary.close()
    save()


if __name__ == "__main__":
    import typer

    typer.run(main)
