# coding: utf8
import json
import random
import os
import sys
from pathlib import Path
from mathy.agent.features import (
    FEATURE_FWD_VECTORS,
    FEATURE_BWD_VECTORS,
    FEATURE_LAST_FWD_VECTORS,
    FEATURE_LAST_BWD_VECTORS,
)
from mathy.core.expressions import MathTypeKeys


def iterate_jsonl(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)


def transform(example):
    """xform to text for JQ to test parser"""
    return {"text": example["before"]}


def transform_3_vectors_to_9(example):
    inputs = example["inputs"]
    pad_value = MathTypeKeys["empty"]
    context_pad_value = [pad_value, pad_value, pad_value]
    # Convert single context vectors to double extract window vectors
    for name in [
        FEATURE_FWD_VECTORS,
        FEATURE_BWD_VECTORS,
        FEATURE_LAST_FWD_VECTORS,
        FEATURE_LAST_BWD_VECTORS,
    ]:
        context_vectors = []
        vectors = inputs[name]
        # Expand the context by doing a window of the window (thanks @honnibal for this trick)
        vectors_len = len(vectors)
        for i, vector in enumerate(vectors):
            # Verify we're not working with already converted observations
            assert len(vector) == 3
            last = context_pad_value if i == 0 else vectors[i - 1]
            next = context_pad_value if i > vectors_len - 2 else vectors[i + 1]
            context_vectors.append(last + vector + next)
        inputs[name] = context_vectors

    return example


output_path = Path("./")


def transform_tweets(archive_name):
    """Transform all the tweets in an archive file into some other format."""
    archive_path = Path(archive_name)
    if not archive_path.is_file():
        raise ValueError(f"archive {archive_name} does not exist!")
    out_file = str(output_path / archive_path.name)
    seen_input = set()
    count = 0
    with open(out_file, "w") as file:
        for example in iterate_jsonl(archive_name):
            out_example = transform(example)
            text = out_example["text"]
            if text in seen_input:
                continue
            count = count + 1
            seen_input.add(text)
            file.write(json.dumps(out_example) + "\n")
            if count % 10000 == 0:
                print(f" - {count} examples written")
    print(f"wrote {count} items to {out_file}")


if __name__ == "__main__":
    archive_name = sys.argv[1]
    transform_tweets(archive_name)
