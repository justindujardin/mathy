import shutil
import tempfile
from pathlib import Path

from mathy.reformer import (
    MathyReformer,
    MathyReformerConfig,
    MathyVocab,
    ReformerLMConfig,
    load_dataset,
    get_batch_loss,
)


train_file = str(Path(__file__).parent / "fixtures/reformer_dataset.txt")


def test_reformer_lm():
    input_folder = tempfile.mkdtemp()
    vocab = MathyVocab()
    config = MathyReformerConfig(
        folder=input_folder,
        train_file=train_file,
        reformer=ReformerLMConfig(num_tokens=vocab.vocab_len),
    )
    reformer: MathyReformer = MathyReformer(
        config=config, vocab=vocab,
    )
    dataset = load_dataset(config.train_file, config.reformer.max_seq_len, reformer)
    X = dataset[0][:5]
    Y = dataset[1][:5]
    batch = list(reformer.net.ops.multibatch(reformer.config.batch_size, X, Y))[0]
    loss = get_batch_loss(reformer, batch)
    assert float(loss) > 0.0
    shutil.rmtree(input_folder)
