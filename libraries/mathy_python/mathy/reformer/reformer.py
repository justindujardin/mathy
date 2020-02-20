import gzip
import os
import random
import string
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import typer
from reformer_pytorch import ReformerLM, Recorder
from torch.nn import functional as F
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from wasabi import msg

VOCAB = ["", " ", "\t", "\n"] + list(
    ".+-/^*()[]-?01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
PAD_TOKEN = VOCAB.index("")
CHAR_TO_INT = {char: index for index, char in enumerate(VOCAB)}
VOCAB_LEN = len(VOCAB)
UNK_TOKEN_STR = "<UNK>"
SEQ_LEN = 128
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 128
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 3e-4
SAVE_EVERY = 50
VALIDATE_EVERY = 100
PRINT_EVERY = 25
GENERATE_EVERY = 100
ATTENTION_EVERY = 50
USE_CUDA = False
USE_PROFILER = True

print(f"Use CUDA: {USE_CUDA}")
print(f"Batch size: {BATCH_SIZE}")
print(f"CPU Profiling: {USE_PROFILER}")
if USE_PROFILER:
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

DatasetTuple = Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, int]]]


# For writing to tensorboard
MODEL_NAME = "test_tb.torch"

WRITER = SummaryWriter(f"training/{os.path.basename(MODEL_NAME)}", flush_secs=30)


def decode_text(tokens: torch.Tensor) -> str:
    """Decode a list of integer tensors to produce a string"""
    output: List[str] = []
    for token in tokens.tolist():
        token_index = int(token)
        if token_index >= VOCAB_LEN:
            output.append(UNK_TOKEN_STR)
        else:
            output.append(VOCAB[token_index])
    return "".join(output)


def encode_text(text: str, pad_length: int) -> torch.Tensor:
    """Encode text into a list of indices in the vocabulary"""
    values = [CHAR_TO_INT[c] for c in text]
    while len(values) < pad_length:
        values.append(PAD_TOKEN)
    return torch.from_numpy(np.asarray(values, dtype=np.uint8)).long()


def load_dataset(file_name: str, pad_length: int) -> DatasetTuple:
    """Load a dataset where question/answer pairs are separated by newlines, and
    pad the outputs to match the transformer sequence length."""
    with open(file_name) as f:
        lines = f.readlines()
    in_lines: List[torch.Tensor] = []
    out_lines: List[Tuple[torch.Tensor, int]] = []
    for i, l in tqdm(enumerate(lines), desc=f"loading dataset: {file_name}"):
        encoded = encode_text(l, pad_length)
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

    outputs: List[Tuple[torch.Tensor, int]]
    inputs: List[torch.Tensor]

    def __init__(self, data: DatasetTuple):
        super().__init__()
        self.inputs = data[0]
        self.outputs = data[1]

    def __getitem__(self, index):
        rand_start = torch.randint(0, len(self.inputs), (1,))
        input_seq = self.inputs[rand_start].long()
        output_seq = self.outputs[rand_start][0].long()
        output_len = self.outputs[rand_start][1]
        if USE_CUDA:
            input_seq = input_seq.cuda()
            output_seq = output_seq.cuda()
        return input_seq, (output_seq, output_len)

    def __len__(self):
        return len(self.inputs)


def evaluate_model(
    model: ReformerLM, dataset: ProblemAnswerDataset
) -> Tuple[float, float]:
    """Evaluate a model on a dataset and return a tuple of the total number
    of problems evaluated, the number answered correctly, and the total loss"""
    model.eval()
    loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    loss: float = 0.0
    correct: int = 0
    total: int = len(dataset)
    print_max = 10
    printed = 0
    with torch.no_grad():
        for batch_with_labels in tqdm(loader, desc="evaluating model"):
            # Check correct/incorrect answers
            batch, batch_labels = batch_with_labels
            prediction = model(batch)
            answer: Any
            for X, label, answer in zip(batch, batch_labels[0], prediction):
                expected = decode_text(label).replace("\n", "")
                # argmax resolves the class probs to ints, and squeeze removes extra dim
                answer = decode_text(answer.argmax(-1).squeeze())
                if "\n" in answer:
                    answer = answer[0 : answer.index("\n")]
                if printed < print_max:
                    printed += 1
                    question = decode_text(X).replace("\n", "")
                    print_fn = msg.good if expected == answer else msg.fail
                    op = "==" if expected == answer else "!="
                    msg.info(f"Question: {question}")
                    msg.info(f"Answer  : {expected}")
                    print_fn(f"Model   : {answer}")
                if answer == expected:
                    correct += 1
            loss += get_batch_loss(
                model, batch_with_labels, prediction=prediction
            ).item()
    ratio = correct / total
    msg.divider(f"{int(ratio * 100)}% correct ({correct}/{total})")
    return loss, ratio


def get_batch_loss(
    model,
    data: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    prediction: torch.Tensor = None,
):
    x, y = data
    # Allow passing a prediction if it's already known to avoid duplicate
    # model calls which generally are expensive.
    if prediction is None:
        prediction = model(x)
    label = y[0]
    label_len = int(y[1].max())
    label = label.narrow(1, 0, label_len)
    pred = prediction.narrow(1, 0, label_len)
    loss = cross_entropy(pred.transpose(1, 2), label, reduction="mean")
    return loss


def train(
    model: ReformerLM,
    optimizer: torch.optim.Adam,
    epochs: int,
    train_file: str,
    eval_file: str,
    generalize_file: str = None,
):

    data_train = load_dataset(train_file, SEQ_LEN)
    data_val = load_dataset(eval_file, SEQ_LEN)
    train_dataset = ProblemAnswerDataset(data_train)
    val_dataset = ProblemAnswerDataset(data_val)
    train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
    if generalize_file is not None:
        data_gen = load_dataset(generalize_file, SEQ_LEN)
        gen_dataset = ProblemAnswerDataset(data_gen)
        gen_loader = cycle(DataLoader(gen_dataset, batch_size=BATCH_SIZE))
    with tqdm(total=epochs, desc="Training") as pbar:

        for i in range(epochs):
            model.epoch += 1
            model.train()

            for __ in range(GRADIENT_ACCUMULATE_EVERY):
                loss = get_batch_loss(model, next(train_loader))
                loss.backward()

            WRITER.add_scalar("loss/train", loss, model.epoch)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            if i % ATTENTION_EVERY == 0:
                r_model = Recorder(model)
                with torch.no_grad():
                    batch_with_labels = next(train_loader)
                    # Check correct/incorrect answers
                    batch, batch_labels = batch_with_labels
                    # Clear recorded attentions
                    r_model.clear()
                    prediction = r_model(batch)
                    answer: Any
                    X = batch[0]
                    label = batch_labels[0][0]
                    answer = prediction[0]
                    input_text = decode_text(X)
                    input_len = input_text.index("\n")
                    expected = decode_text(label).replace("\n", "")
                    # argmax resolves the class probs to ints, and squeeze removes extra dim
                    answer = decode_text(answer.argmax(-1).squeeze())
                    if "\n" in answer:
                        answer = answer[0 : answer.index("\n")]
                    layers = r_model.recordings[0][0]["attn"]
                    for i, attn_layer in enumerate(layers):
                        for j, attn_head in enumerate(attn_layer):
                            title = f"layer: {i} head: {j} correct: {answer == expected} answer: {expected} model: {answer}"
                            figure = plot_attention(
                                attn_head,
                                input_text,
                                input_text,
                                input_len,
                                title,
                                return_figure=True,
                            )
                            WRITER.add_figure(
                                title, figure, global_step=model.epoch,
                            )
            pbar.desc = f"loss: {loss.item()}"
            if i % PRINT_EVERY == 0:
                print(f"Step: {model.epoch}")

            if i % SAVE_EVERY == 0:
                if i > 0:
                    print(f"Saving checkpoint: {MODEL_NAME}")
                    torch.save(
                        {
                            "epoch": model.epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        MODEL_NAME,
                    )
            if i % VALIDATE_EVERY == 0:
                eval_loss, eval_win_pct = evaluate_model(model, val_dataset)
                print(f"Loss (train): {loss.item()}")
                print(f"Loss (eval) : {eval_loss}")
                if generalize_file:
                    gen_loss, gen_correct = evaluate_model(model, gen_dataset)
                    WRITER.add_scalar("loss/generalize", gen_loss, model.epoch)
                    WRITER.add_scalar("correct/generalize", gen_correct, model.epoch)
                    print(f"Loss (gen)  : {gen_loss}")
                WRITER.add_scalar("loss/eval", eval_loss, model.epoch)
                WRITER.add_scalar("correct/eval", eval_win_pct, model.epoch)

            pbar.update(1)


# From: https://www.tensorflow.org/tutorials/text/nmt_with_attention
def plot_attention(
    attention: torch.Tensor,
    input_one: str,
    input_two: str,
    seq_len: int,
    title: str,
    return_figure: bool = False,
):
    fig = plt.figure(figsize=(seq_len, seq_len))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention[0:seq_len, 0:seq_len], cmap="viridis")

    fontdict = {"fontsize": 14}

    plt.title(title)

    ax.set_xticklabels([""] + list(input_one), fontdict=fontdict)
    ax.set_yticklabels([""] + list(input_two), fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    if return_figure is False:
        plt.show()
    else:
        return fig


def main():
    # instantiate model
    model = ReformerLM(
        dim=128,
        depth=1,
        max_seq_len=SEQ_LEN,
        num_tokens=VOCAB_LEN,
        bucket_size=16,
        heads=2,
        n_hashes=2,
        ff_chunks=0,
        lsh_dropout=0.1,
    )
    # WRITER.add_graph(model, torch.zeros(4, SEQ_LEN).long())
    if USE_CUDA:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if os.path.exists(MODEL_NAME):
        print(f"Loading existing model: {MODEL_NAME}")
        dev = "cpu" if USE_CUDA is False else "gpu"
        checkpoint = torch.load(MODEL_NAME, map_location=torch.device(dev))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.epoch = checkpoint.get("epoch", 0)
        print(f"Loaded model (epoch={model.epoch})")
    else:
        model.epoch = 0

    train(
        model, optimizer, 1000, "hoarding/tiny.train.txt", "hoarding/ltp_hard.eval.txt"
    )
    # train(
    #     model,
    #     optimizer,
    #     1000,
    #     "like.train.txt",
    #     "like.eval.txt",
    #     "like.generalization.txt",
    # )


if __name__ == "__main__":
    typer.run(main)
