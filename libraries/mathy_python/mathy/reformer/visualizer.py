import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import ticker
from reformer_pytorch import Recorder
import torch
from thinc.api import Ops, get_current_ops
from .reformer import (
    DatasetTuple,
    MathyReformer,
    MathyReformerConfig,
    MathyVocab,
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
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title)
    ax.matshow(attention[0:seq_len, 0:seq_len], cmap="plasma")
    ax.set_xticklabels([""] + list(text), fontdict=font_opts)
    ax.set_yticklabels([""] + list(text), fontdict=font_opts)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    return True


def evaluate_model_attention(model: MathyReformer, dataset: DatasetTuple) -> None:
    """Evaluate a model on a dataset and return a tuple of the total number
    of problems evaluated, the number answered correctly, and the total loss """
    assert isinstance(model.reformer, Recorder), "record_attention must be True"
    model.reformer.net.eval()
    ops: Ops = get_current_ops()
    batches = ops.multibatch(1, dataset[0], dataset[1], shuffle=True)
    with torch.no_grad():
        for batch_with_labels in batches:
            batch, batch_labels = batch_with_labels
            batch = torch.from_numpy(ops.asarray(batch))
            # Clear recorded attentions
            model.reformer.clear()
            prediction = model.reformer(batch)
            answer: Any
            X = batch[0]
            label = batch_labels[0][0]
            answer = prediction
            input_text = model.vocab.decode_text(X)
            input_len = input_text.index("\n")
            expected = model.vocab.decode_text(label).replace("\n", "")
            # argmax resolves the class probs to ints, and squeeze removes extra dim
            answer = model.vocab.decode_text(answer.argmax(-1).squeeze())
            if "\n" in answer:
                answer = answer[0 : answer.index("\n")]
            question = input_text.replace("\n", "")
            print(f"Question: {question}")
            print(f"Answer  : {expected}")
            print(f"Model   : {answer}")
            correct_str = "CORRECT" if expected == answer else "WRONG"
            # The like terms attention tends to be informative on head/layer 0
            attn_head = model.reformer.recordings[0][0]["attn"][0][0]
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
    model: MathyReformer = MathyReformer(
        config=config, vocab=vocab, must_exist=True, record_attention=True
    )
    print(f"Folder: {config.folder}")
    print(f"Config: {json.dumps(model.config.dict(), indent=2)}")
    print("Loading dataset")

    dataset = load_dataset(config.eval_file, config.reformer.max_seq_len, model.vocab)
    print("Loaded dataset")
    evaluate_model_attention(model, dataset)
