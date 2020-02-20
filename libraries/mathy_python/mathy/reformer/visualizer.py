import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from reformer_pytorch import ReformerLM, Recorder
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from wasabi import msg

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .reformer import (
    PAD_TOKEN,
    SEQ_LEN,
    USE_CUDA,
    VOCAB,
    VOCAB_LEN,
    DatasetTuple,
    ProblemAnswerDataset,
    decode_text,
    encode_text,
    load_dataset,
)



def show_head_view(model, tokenizer, sentence_a, sentence_b=None):
    inputs = tokenizer.encode_plus(
        sentence_a, sentence_b, return_tensors="pt", add_special_tokens=True
    )
    input_ids = inputs["input_ids"]
    if sentence_b:
        token_type_ids = inputs["token_type_ids"]
        attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        sentence_b_start = token_type_ids[0].tolist().index(1)
    else:
        attention = model(input_ids)[-1]
        sentence_b_start = None
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    head_view(attention, tokens, sentence_b_start)


def evaluate_model(model: ReformerLM, dataset: ProblemAnswerDataset) -> float:
    """Evaluate a model on a dataset and return a tuple of the total number
    of problems evaluated, the number answered correctly, and the total loss"""
    model.eval()
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size)
    loss: float = 0.0
    correct: int = 0
    total: int = len(dataset)
    with torch.no_grad():
        for batch_with_labels in loader:
            # Check correct/incorrect answers
            batch, batch_labels = batch_with_labels
            # Clear recorded attentions
            model.clear()
            prediction = model(batch)
            answer: Any

            for batch_i, X, label, answer in zip(
                list(range(batch_size)), batch, batch_labels[0], prediction
            ):
                input_text = decode_text(X)
                input_len = input_text.index("\n")
                print(input_len)
                expected = decode_text(label).replace("\n", "")
                # argmax resolves the class probs to ints, and squeeze removes extra dim
                answer = decode_text(answer.argmax(-1).squeeze())
                if "\n" in answer:
                    answer = answer[0 : answer.index("\n")]
                if True:
                    question = input_text.replace("\n", "")
                    print_fn = msg.good if expected == answer else msg.fail
                    op = "==" if expected == answer else "!="
                    msg.info(f"Question: {question}")
                    msg.info(f"Answer  : {expected}")
                    print_fn(f"Model   : {answer}")
                correct += 1
                layers = model.recordings[0][batch_i]["attn"]
                for i, attn_layer in enumerate(layers):
                    for j, attn_head in enumerate(attn_layer):
                        title = f"layer: {i} head: {j} correct: {answer == expected} answer: {expected} model: {answer}"
                        print(title)
                        plot_attention(
                            attn_head, input_text, input_text, input_len, title,
                        )

    pct = int(correct / total * 100)
    msg.divider(f"{pct}% correct ({correct}/{total})")
    return loss


if __name__ == "__main__":

    file_name: str = "hoarding/like_terms_prediction.torch"
    model = ReformerLM(
        dim=512,
        depth=4,
        max_seq_len=SEQ_LEN,
        num_tokens=VOCAB_LEN,
        bucket_size=16,
        heads=2,
        n_hashes=2,
        ff_chunks=8,
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
    dataset = ProblemAnswerDataset(load_dataset("hoarding/ltp_hard.eval.txt", SEQ_LEN))
    evaluate_model(model, dataset)
