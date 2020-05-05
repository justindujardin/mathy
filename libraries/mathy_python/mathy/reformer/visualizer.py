import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from reformer_pytorch import ReformerLM, Recorder
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .reformer import (
    SEQ_LEN,
    USE_CUDA,
    VOCAB_LEN,
    ProblemAnswerDataset,
    decode_text,
    load_dataset,
)


# From: https://www.tensorflow.org/tutorials/text/nmt_with_attention
def plot_attention(
    attention: torch.Tensor,
    input_one: str,
    input_two: str,
    seq_len: int,
    title: str,
    answer: int,
) -> bool:

    if seq_len < 35:
        fig_size = 6
        font_size = 12
    else:
        fig_size = 12
        font_size = 8

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.matshow(attention[0:seq_len, 0:seq_len], cmap="viridis")

    fontdict = {"fontsize": font_size}

    ax.set_xticklabels([""] + list(input_one), fontdict=fontdict)
    ax.set_yticklabels([""] + list(input_two), fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    return True


def evaluate_model_attention(model: ReformerLM, dataset: ProblemAnswerDataset) -> None:
    """Evaluate a model on a dataset and return a tuple of the total number
    of problems evaluated, the number answered correctly, and the total loss"""
    model.eval()
    loader = DataLoader(dataset, batch_size=1)
    for batch_with_labels in loader:
        batch, batch_labels = batch_with_labels
        # Clear recorded attentions
        model.clear()
        prediction = model(batch)
        answer: Any
        X = batch[0]
        label = batch_labels[0][0]
        answer = prediction
        input_text = decode_text(X)
        input_len = input_text.index("\n")
        expected = decode_text(label).replace("\n", "")
        # argmax resolves the class probs to ints, and squeeze removes extra dim
        answer = decode_text(answer.argmax(-1).squeeze())
        question = input_text.replace("\n", "")
        print(f"Question: {question}")
        print(f"Answer  : {expected}")
        print(f"Model   : {answer}")
        correct_str = "CORRECT" if expected == answer else "WRONG"
        # The like terms attention tends to be informative on head/layer 0
        attn_head = model.recordings[0][0]["attn"][0][0]
        title = f"expected: {expected} model:{answer} - {correct_str}"
        plot_attention(
            attention=attn_head,
            input_one=input_text,
            input_two=input_text,
            seq_len=input_len,
            title=title,
            answer=int(answer),
        )


if __name__ == "__main__":
    file_name: str = "training/reformer/ltp_only_hard.torch"
    model = ReformerLM(
        dim=512,
        depth=2,
        max_seq_len=SEQ_LEN,
        num_tokens=VOCAB_LEN,
        bucket_size=64,
        heads=4,
        n_hashes=4,
        ff_chunks=0,
        lsh_dropout=0.1,
    )
    if USE_CUDA:
        model.cuda()
    if not os.path.exists(file_name):
        raise ValueError(f"Model does not exist: {file_name}")
    print(f"Loading existing model: {file_name}")
    dev = "cpu" if USE_CUDA is False else "cuda"
    checkpoint = torch.load(file_name, map_location=torch.device(dev))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = Recorder(model)
    dataset = ProblemAnswerDataset(
        load_dataset("training/reformer/ltp_hard.eval.txt", SEQ_LEN)
    )
    evaluate_model_attention(model, dataset)
