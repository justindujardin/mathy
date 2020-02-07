import gzip
import os
import random
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import typer
from reformer_pytorch import ReformerLM
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from wasabi import msg

VOCAB = ["", " ", "\t", "\n"] + list(
    ".+-/^*()[]-?01234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
)
PAD_TOKEN = VOCAB.index("")
INT_TO_CHAR = {index: char for index, char in enumerate(VOCAB)}
CHAR_TO_INT = {char: index for index, char in enumerate(VOCAB)}
VOCAB_LEN = len(VOCAB)

MODEL_NAME = "model.torch"
OPTIMIZER_NAME = f"{MODEL_NAME}.optimizer"
UNK_TOKEN_STR = "<UNK>"
NUM_BATCHES = int(1e5)
SEQ_LEN = 128
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 3e-4
SAVE_EVERY = 100
VALIDATE_EVERY = 200
GENERATE_EVERY = 100

USE_CUDA = False
print(f"Use CUDA: {USE_CUDA}")


def decode_token(token):
    token_index = int(token)
    if token_index >= VOCAB_LEN:
        token = UNK_TOKEN_STR
    else:
        token = INT_TO_CHAR[token_index]
    return token


def decode_tokens(tokens):
    output = "".join(list(map(decode_token, tokens)))
    return output


def cycle(loader):
    while True:
        for data in loader:
            yield data


def pad_array(in_list: List[Any], max_length: int, value: Any = 0) -> List[Any]:
    while len(in_list) < max_length:
        in_list.append(value)
    return in_list


def encode_input(text: str) -> torch.Tensor:
    values = [CHAR_TO_INT[c] for c in text]
    values = pad_array(values, SEQ_LEN, PAD_TOKEN)
    return torch.from_numpy(np.asarray(values, dtype=np.uint8)).long()


DatasetTuple = Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, int]]]


def load_dataset(file_name: str) -> DatasetTuple:
    with open(file_name) as f:
        lines = f.readlines()
    in_lines: List[torch.Tensor] = []
    out_lines: List[Tuple[torch.Tensor, int]] = []
    for i, l in tqdm(enumerate(lines), desc=f"loading dataset: {file_name}"):
        if i % 2 == 0:
            in_lines.append(encode_input(l))
        else:
            out_lines.append((encode_input(l), len(l)))

    assert len(in_lines) == len(out_lines), "in/out files must have 1-to-1 line mapping"

    return in_lines, out_lines


class ProblemAnswerDataset(Dataset):
    """A dataset loader that reads newline delimited problem/answer examples
    from a file """

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


def main(
    train_file: str = "like.train.txt",
    eval_file: str = "like.eval.txt",
    generalize_file: str = "like.generalization.txt",
):
    current = os.path.dirname(__file__)
    data_train = load_dataset(train_file)
    data_val = load_dataset(eval_file)
    train_dataset = ProblemAnswerDataset(data_train)
    val_dataset = ProblemAnswerDataset(data_val)
    train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
    val_loader = cycle(DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE))
    if generalize_file:
        data_gen = load_dataset(generalize_file)
        gen_dataset = ProblemAnswerDataset(data_gen)
        gen_loader = cycle(DataLoader(gen_dataset, batch_size=EVAL_BATCH_SIZE))

    def get_batch_loss(
        model, data: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ):
        x, y = data
        pred = model(x)
        label = y[0]
        label_len = int(y[1].max())
        label = label.narrow(1, 0, label_len)
        pred = pred.narrow(1, 0, label_len)
        loss = F.cross_entropy(pred.transpose(1, 2), label, reduction="mean")
        return loss

    model = ReformerLM(
        dim=128,
        depth=2,
        max_seq_len=SEQ_LEN,
        num_tokens=VOCAB_LEN,
        heads=4,
        bucket_size=64,
        n_hashes=2,
        ff_chunks=8,
        lsh_dropout=0.1,
    )
    if USE_CUDA:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if os.path.exists(MODEL_NAME):
        print(f"Loading existing model: {MODEL_NAME}")
        checkpoint = torch.load(MODEL_NAME)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    with tqdm(total=NUM_BATCHES, desc="Training") as pbar:
        for i in range(NUM_BATCHES):
            pbar.update(1)
            model.train()

            for __ in range(GRADIENT_ACCUMULATE_EVERY):
                loss = get_batch_loss(model, next(train_loader))
                loss.backward()

            pbar.desc = f"training loss: {loss.item()}"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()

            if i % VALIDATE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    eval_loss = get_batch_loss(model, next(val_loader))
                    print(f"eval loss:           {eval_loss.item()}")
                    if generalize_file:
                        gen_loss = get_batch_loss(model, next(gen_loader))
                        print(f"generalization loss: {gen_loss.item()}")

            if i % GENERATE_EVERY == 0:
                model.eval()
                if i > 0:
                    print(f"Saving checkpoint: {MODEL_NAME}")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        MODEL_NAME,
                    )
                with torch.no_grad():
                    batch_with_labels = random.choice(val_dataset)  # type:ignore
                    X: torch.Tensor = batch_with_labels[0]
                    Y: Tuple[torch.Tensor, torch.Tensor] = batch_with_labels[1]
                    label = Y[0]
                    output_str = ""
                    question = decode_tokens(X).replace("\n", "")
                    expected = decode_tokens(label).replace("\n", "")
                    answer = model(X[None, :])
                    # argmax resolves the class probs to ints, and squeeze removes extra dim
                    answer = decode_tokens(answer.argmax(-1).squeeze())
                    if "\n" in answer:
                        answer = answer[0 : answer.index("\n")]
                    print_fn = msg.good if expected == answer else msg.fail
                    op = "==" if expected == answer else "!="
                    msg.info(f"Question:  {question}")
                    print_fn(f"Answer:    {expected}    Predicted: {answer}")


if __name__ == "__main__":
    typer.run(main)
