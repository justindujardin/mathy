from ..environment_state import INPUT_EXAMPLES_FILE_NAME
from pathlib import Path
import ujson
import tempfile
from shutil import copyfile
import os


class MathExperience:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.long_term = []
        self.short_term = []
        if self._load_experience() is not False:
            pass

    def add_batch(self, new_examples):
        """Add a batch of experience from observations and save short/long-term memory to disk"""
        self.long_term.extend(self.short_term)
        self.short_term = new_examples
        self._save_experience()

    def _load_experience(self):
        file_path = Path(self.model_dir) / INPUT_EXAMPLES_FILE_NAME
        if not file_path.is_file():
            return False
        examples = []
        with file_path.open("r", encoding="utf8") as f:
            for line in f:
                ex = ujson.loads(line)
                examples.append(ex)
        self.long_term = examples
        return True

    def _save_experience(self):
        model_dir = Path(self.model_dir)
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)

        all_experience = self.long_term + self.short_term

        # Write to local file then copy over (don't thrash virtual file systems like GCS)
        _, tmp_file = tempfile.mkstemp()
        with Path(tmp_file).open("w", encoding="utf-8") as f:
            for line in all_experience:
                f.write(ujson.dumps(line, escape_forward_slashes=False) + "\n")

        out_file = model_dir / INPUT_EXAMPLES_FILE_NAME
        copyfile(tmp_file, str(out_file))
        os.remove(tmp_file)
        return str(out_file)

    def save_training_examples_tfrecord(self):
        model_dir = Path(self.model_dir)
        if not model_dir.is_dir():
            model_dir.mkdir(parents=True, exist_ok=True)

        # Write to local file then copy over (don't thrash virtual file systems like GCS)
        _, tmp_file = tempfile.mkstemp()

        writer = tf.io.TFRecordWriter(tmp_file)
        for example in self.long_term:
            ex = tf.train.SequenceExample()
            # A non-sequential feature of our example
            sequence_length = len(sequence)
            ex.context.feature["length"].int64_list.value.append(sequence_length)
            # Feature lists for the two sequential features of our example
            fl_tokens = ex.feature_lists.feature_list["tokens"]
            fl_labels = ex.feature_lists.feature_list["labels"]
            for token, label in zip(sequence, labels):
                fl_tokens.feature.add().int64_list.value.append(token)
                fl_labels.feature.add().int64_list.value.append(label)

            ex = make_example(sequence, label_sequence)
            writer.write(ex.SerializeToString())
        writer.close()

        out_file = model_dir / INPUT_EXAMPLES_FILE_NAME
        copyfile(tmp_file, str(out_file))
        os.remove(tmp_file)
        return str(out_file)

