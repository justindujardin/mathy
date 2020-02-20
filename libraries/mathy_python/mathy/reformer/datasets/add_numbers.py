import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import typer
from tqdm.auto import tqdm
from mathy.problems import rand_number


def generate_problems(
    file_base: str, number: int, exclude: Optional[Set[str]] = None, eval: bool = False
):
    inputs_file_name = f"{file_base}.inputs.txt"
    outputs_file_name = f"{file_base}.outputs.txt"
    if exclude is None:
        exclude = set()
    skips = 0
    skip_threshold = 1200
    problems = 0

    with Path(outputs_file_name).open("w") as outputs_file:
        with Path(inputs_file_name).open("w") as inputs_file:
            with tqdm(total=number, mininterval=0.25, desc=file_base) as pbar:
                while problems < number:
                    one = rand_number()
                    two = rand_number()
                    text = f"{one}+{two}"
                    answer = f"{one + two}"
                    if text in exclude:
                        skips += 1
                        if skips >= skip_threshold:
                            raise ValueError(
                                f"Failed to generate more unique problems after {skips} tries!"
                            )
                            continue
                    exclude.add(text)
                    outputs_file.write(f"{answer}\n")
                    inputs_file.write(f"{text}\n")
                    pbar.update(1)
                    problems += 1


def main(
    name: str,
    train_size: int = 100 * 1000,
    eval_size: int = 10 * 1000,
    include_eval: bool = True,
    include_generalization: bool = True,
):
    current = os.path.dirname(__file__)
    train_file = os.path.join(current, f"{name}.train")
    eval_file = os.path.join(current, f"{name}.eval")
    generalization_file = os.path.join(current, f"{name}.generalization")

    seen_texts: Set[str] = set()

    print(f"Generating train dataset with {train_size} examples...")
    generate_problems(train_file, train_size, seen_texts, eval=True)
    if include_eval:
        print(f"Generating eval dataset with {eval_size} examples...")
        generate_problems(eval_file, eval_size, seen_texts, eval=True)

    if include_generalization:
        print(f"Generating generalization dataset with {eval_size} examples...")
        generate_problems(generalization_file, eval_size, seen_texts)


if __name__ == "__main__":
    typer.run(main)
