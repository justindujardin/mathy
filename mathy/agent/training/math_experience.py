import os
import tempfile
from pathlib import Path
from shutil import copyfile
import random
import ujson

from ...environment_state import INPUT_EXAMPLES_FILE_NAME


def balanced_reward_experience_samples(examples_pool, max_items: int):
    """Long-term memory sampling function that tries to return a roughly 
    equal distribution of positive/negative reward examples for training.
    If there are not enough examples of a particular type (i.e. positive/negative)
    examples from the other class will be filled in until max_items is met
    or all long-term memory examples are exhausted.

    This sampling method is inspired by UNREAL agent (https://arxiv.org/pdf/1611.05397.pdf)
    and their insights that "animals dream about positively or negatively rewarding events 
    more frequently" which were gathered from (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3815616/)
    which found this by observing the brain activity of animals while they slept.    
    """
    overflow_examples = []
    positive_examples = []
    negative_examples = []
    shuffled = examples_pool[:]
    half_examples = int(max_items / 2)
    random.shuffle(shuffled)
    for example in examples_pool:
        reward = example["reward"]
        if reward < 0.0 and len(negative_examples) < half_examples:
            negative_examples.append(example)
        elif reward > 0.0 and len(positive_examples) < half_examples:
            positive_examples.append(example)
        else:
            overflow_examples.append(example)
        if (
            len(positive_examples) == half_examples
            and len(negative_examples) == half_examples
            and len(overflow_examples) > half_examples
        ):
            break
    out_examples = positive_examples + negative_examples
    overflow_count = 0
    while len(out_examples) < max_items and len(overflow_examples) > 0:
        overflow_count += 1
        out_examples.append(overflow_examples.pop())
    print(
        f"[ltm] Sampled ({len(positive_examples)}) positive"
        f", ({len(negative_examples)}) negative"
        f", and ({overflow_count}) overflow examples"
    )
    return out_examples


class MathExperience:
    #
    #
    # TODO: Add a concept of a second store of long-term memories from a "mentor"
    #       and use that experience to fill up to a minimum buffer size when training
    #       new models from pretrained existing models. This way you can gather 50
    #       examples of a new problem type and mix in (max - 50) examples from the
    #       mentor's long-term experience.
    # TODO: categorize experience by problem type and/or complexity
    #       so that we can do things like evenly distribute training across types
    #       or sample from long-term memories evenly across all known problem types.
    #       I think this will be useful for generalization because we can force the
    #       model to maintain a balance of training inputs across everything it knows.
    #
    def __init__(self, model_dir, short_term_size=64):
        self.model_dir = model_dir
        self.long_term = []
        self.short_term = []
        self.short_term_size = short_term_size
        if self._load_experience() is not False:
            pass

    @property
    def count(self):
        """Returns the count of all experience"""
        return len(self.short_term) + len(self.long_term)

    def all(self):
        """Returns a concatenation of short-term and long-term memory arrays"""
        return self.short_term + self.long_term

    def add_batch(self, new_examples):
        """Add a batch of experience from observations and save short/long-term memory to disk"""
        new_size = len(self.short_term) + len(new_examples)
        # When the short-term buffer fills up, dump the oldest items to long-term
        if new_size >= self.short_term_size:
            to_remove = new_size - self.short_term_size
            self.long_term.extend(self.short_term[:to_remove])
            self.short_term = self.short_term[to_remove:]
        self.short_term.extend(new_examples)
        self._save_experience()

    def _load_experience(self):
        # Try to match a specified file first
        file_path = Path(self.model_dir)
        if not file_path.is_file():
            # Check for a folder with the inputs file inside of it
            file_path = file_path / INPUT_EXAMPLES_FILE_NAME
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
        fd, tmp_file = tempfile.mkstemp()
        with Path(tmp_file).open("w", encoding="utf-8") as f:
            for line in all_experience:
                f.write(ujson.dumps(line, escape_forward_slashes=False) + "\n")

        out_file = model_dir / INPUT_EXAMPLES_FILE_NAME
        if out_file.is_file():
            copyfile(str(out_file), f"{str(out_file)}.bak")
        copyfile(tmp_file, str(out_file))
        os.remove(tmp_file)
        os.close(fd)
        return str(out_file)
