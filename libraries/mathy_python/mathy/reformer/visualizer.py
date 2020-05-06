import json
from typing import Any

import matplotlib
from reformer_pytorch import Recorder
import torch
from torch.utils.data import DataLoader

from .reformer import (
    MathyReformer,
    MathyReformerConfig,
    MathyVocab,
    ProblemAnswerDataset,
    ReformerLMConfig,
    load_dataset,
)

# From: https://www.tensorflow.org/tutorials/text/nmt_with_attention
def plot_attention(
    attention: torch.Tensor, text: str, seq_len: int, title: str, answer: int,
) -> bool:
    big_problem_len = 35
    fig_size = 6 if seq_len < big_problem_len else 12
    font_size = 10 if seq_len < big_problem_len else 8
    font_opts = {"fontsize": font_size}
    fig = matplotlib.pyplot.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.matshow(attention[0:seq_len, 0:seq_len], cmap="plasma")
    ax.set_xticklabels([""] + list(text), fontdict=font_opts)
    ax.set_yticklabels([""] + list(text), fontdict=font_opts)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    matplotlib.pyplot.show()
    return True


def evaluate_model_attention(recorder: Recorder, dataset: ProblemAnswerDataset) -> None:
    """Evaluate a model on a dataset and return a tuple of the total number
    of problems evaluated, the number answered correctly, and the total loss"""
    recorder.net.eval()
    loader = DataLoader(dataset, batch_size=1)
    for batch_with_labels in loader:
        batch, batch_labels = batch_with_labels
        # Clear recorded attentions
        model.clear()
        prediction = recorder(batch)
        answer: Any
        X = batch[0]
        label = batch_labels[0][0]
        answer = prediction
        input_text = model.net.vocab.decode_text(X)
        input_len = input_text.index("\n")
        expected = model.net.vocab.decode_text(label).replace("\n", "")
        # argmax resolves the class probs to ints, and squeeze removes extra dim
        answer = model.net.vocab.decode_text(answer.argmax(-1).squeeze())
        if "\n" in answer:
            answer = answer[0 : answer.index("\n")]
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
            text=input_text,
            seq_len=input_len,
            title=title,
            answer=int(answer),
        )


if __name__ == "__main__":
    vocab = MathyVocab()
    config = MathyReformerConfig(
        folder="training/reformer/lte_latest",
        eval_file="training/reformer/like_terms_prediction.eval.txt",
        reformer=ReformerLMConfig(num_tokens=vocab.vocab_len),
    )
    reformer: MathyReformer = MathyReformer(
        config=config, vocab=vocab, must_exist=True,
    )
    print(f"Folder: {config.folder}")
    print(f"Config: {json.dumps(reformer.config.dict(), indent=2)}")
    model = Recorder(reformer)
    dataset_loader = load_dataset(
        config.eval_file, config.reformer.max_seq_len, reformer.vocab
    )
    dataset = ProblemAnswerDataset(config, dataset_loader)
    evaluate_model_attention(model, dataset)
