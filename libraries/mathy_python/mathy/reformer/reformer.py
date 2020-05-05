import json
import time
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from reformer_pytorch import ReformerLM
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from wasabi import msg

# TODO: Replace with thinc to make swapping frameworks easier
TensorType = torch.Tensor


class MathyReformerConfig(BaseModel):
    folder: str
    train_file: str
    eval_file: str
    seq_len: int = 128
    batch_size: int = 512
    eval_batch_size: int = 128
    accumulate_every: int = 4
    learning_rate: float = 3e-4
    save_every: int = 100
    validate_every: int = 100
    print_every: int = 10
    use_cuda: bool = False

    # ReformerLM config
    reformer_dim: int = 512
    reformer_depth: int = 2
    reformer_bucket_size: int = 64
    reformer_heads: int = 4
    reformer_n_hashes: int = 4
    reformer_ff_chunks: int = 0
    reformer_lsh_dropout: float = 0.1


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

    def to_int(self, char: str):
        return self.char_to_int[char]

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
        if pad_length is not None:
            padding = [self.pad_token] * (pad_length - len(text))
        else:
            padding = []
        values = [self.char_to_int[c] for c in text] + padding
        if include_batch:
            values = [values]
        return torch.from_numpy(np.asarray(values, dtype=np.uint8)).long()


class MathyReformer(ReformerLM):
    config: MathyReformerConfig
    vocab: MathyVocab

    def __init__(self, config: MathyReformerConfig, vocab: MathyVocab, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.vocab = vocab


def main(
    folder: str = "training/reformer/dev",
    train_file="like_terms_prediction.train.txt",
    eval_file="like_terms_prediction.eval.txt",
):
    config = MathyReformerConfig(
        folder=folder, train_file=train_file, eval_file=eval_file
    )
    vocab = MathyVocab()
    print(f"Config: {json.dumps(config.dict(), indent=2)}")
    print(f"Batch Size: {config.batch_size}")
    reformer: MathyReformer = MathyReformer(
        config=config,
        vocab=vocab,
        dim=config.reformer_dim,
        depth=config.reformer_depth,
        max_seq_len=config.seq_len,
        num_tokens=vocab.vocab_len,
        bucket_size=config.reformer_bucket_size,
        heads=config.reformer_heads,
        n_hashes=config.reformer_n_hashes,
        ff_chunks=config.reformer_ff_chunks,
        lsh_dropout=config.reformer_lsh_dropout,
    )
    if config.use_cuda:
        reformer.cuda()

    optimizer = torch.optim.Adam(reformer.parameters(), lr=config.learning_rate)
    model = os.path.join(folder, "model.torch")
    if os.path.exists(model):
        print(f"loading model: {model}")
        checkpoint = torch.load(model)
        reformer.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        reformer.epoch = checkpoint.get("epoch", 0)
    else:
        reformer.epoch = 0
    print(f"model epoch: {reformer.epoch}")
    train(
        reformer, config=config, model_name=model, optimizer=optimizer, epochs=3000,
    )


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


class ProblemAnswerDataset(Dataset):
    """A sequence-to-sequence dataset for question/answer pairs.
    
    Provided label is a tuple of two tensors, the first is the max-length padded label
    and the second is a tensor containing the actual label lengths.
    
    The label lengths are used to mask the output when calculating loss."""

    config: MathyReformerConfig
    outputs: List[Tuple[TensorType, int]]
    inputs: List[TensorType]

    def __init__(self, config: MathyReformerConfig, data: DatasetTuple):
        super().__init__()
        self.config = config
        self.inputs = data[0]
        self.outputs = data[1]

    def __getitem__(self, index):
        rand_start = int(torch.randint(0, len(self.inputs), (1,)))
        input_seq = self.inputs[rand_start].long()
        output_seq = self.outputs[rand_start][0].long()
        output_len = self.outputs[rand_start][1]
        if self.config.use_cuda:
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()
        return input_seq, (output_seq, output_len)

    def __len__(self):
        return len(self.inputs)


def evaluate_model(
    model: MathyReformer, dataset: ProblemAnswerDataset
) -> Tuple[float, float]:
    """Evaluate a model on a dataset and return a tuple of the total number
    of problems evaluated, the number answered correctly, and the total loss"""
    model.eval()
    loader = DataLoader(dataset, batch_size=model.config.batch_size)
    loss: float = 0.0
    correct: int = 0
    total: int = len(dataset)
    print_max = 3
    printed = 0
    with torch.no_grad():
        for batch_with_labels in loader:
            # Check correct/incorrect answers
            batch, batch_labels = batch_with_labels
            prediction = model(batch)
            answer: Any
            for X, label, answer in zip(batch, batch_labels[0], prediction):
                expected = model.vocab.decode_text(label).replace("\n", "")
                # argmax resolves the class probs to ints, and squeeze removes extra dim
                answer = model.vocab.decode_text(answer.argmax(-1).squeeze())
                if "\n" in answer:
                    answer = answer[0 : answer.index("\n")]
                if printed < print_max:
                    printed += 1
                    question = model.vocab.decode_text(X).replace("\n", "")
                    outcome = "WRONG" if expected != answer else "RIGHT"
                    print(
                        f"{outcome} | answer: {expected} | model: {answer} | question: {question}"
                    )
                if answer == expected:
                    correct += 1
            loss += get_batch_loss(
                model, batch_with_labels, prediction=prediction
            ).item()
    ratio = correct / total
    print(f"evaluation accuracy: {int(ratio * 100)}% | correct: ({correct}/{total})")
    return loss, ratio


def get_batch_loss(
    model,
    data: Tuple[TensorType, Tuple[TensorType, TensorType]],
    prediction: TensorType = None,
):
    x, y = data
    # Allow passing a prediction to avoid duplicate model calls
    if prediction is None:
        prediction = model(x)
    # y is a tuple of (label, length) to allow unpadding
    label = y[0]
    label_len = int(y[1].max())
    label = label.narrow(1, 0, label_len)
    pred = prediction.narrow(1, 0, label_len)
    loss = cross_entropy(pred.transpose(1, 2), label, reduction="mean")
    return loss


def train(
    model: MathyReformer,
    *,
    model_name: str,
    config: MathyReformerConfig,
    optimizer: torch.optim.Adam,
    epochs: int,
):

    summary = SummaryWriter(os.path.join(config.folder, "tensorboard"), flush_secs=30)
    data_train = load_dataset(config.train_file, config.seq_len, model.vocab)
    data_val = load_dataset(config.eval_file, config.seq_len, model.vocab)
    train_dataset = ProblemAnswerDataset(config, data_train)
    val_dataset = ProblemAnswerDataset(config, data_val)
    train_loader = cycle(DataLoader(train_dataset, batch_size=config.batch_size))
    loss: TensorType = torch.zeros(1)
    for i in range(epochs):
        model.epoch += 1
        model.train()

        step_start = time.time()
        for __ in range(config.accumulate_every):
            loss = get_batch_loss(model, next(train_loader))
            loss.backward()
        print(".", end="", flush=True)
        step_end = time.time()
        summary.add_scalar("metrics/epoch_time", step_end - step_start, model.epoch)
        summary.add_scalar("loss/train", loss, model.epoch)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        optimizer.zero_grad()

        if i % config.print_every == 0:
            print(f"step: {model.epoch} | loss: {loss.item()}")

        if i % config.save_every == 0:
            if i > 0:
                print(
                    f"save: {model_name} | step: {model.epoch} | loss_train: {loss.item()}"
                )
                torch.save(
                    {
                        "epoch": model.epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    model_name,
                )
        if i % config.validate_every == 0:
            eval_loss, eval_win_pct = evaluate_model(model, val_dataset)
            print(f"loss_train: {loss.item()} | loss_eval: {eval_loss}")
            summary.add_scalar("loss/eval", eval_loss, model.epoch)
            summary.add_scalar("correct/eval", eval_win_pct, model.epoch)

    summary.close()


if __name__ == "__main__":
    import typer

    typer.run(main)
