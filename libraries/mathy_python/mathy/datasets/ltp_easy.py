"""Utility for generating question/answer datasets of two terms that are either like or unlike"""
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import typer
from tqdm.auto import tqdm

from mathy import ExpressionParser, MathExpression, TermEx, get_term_ex, terms_are_like
from mathy.problems import (
    MathyTermTemplate,
    gen_simplify_multiple_terms,
    get_rand_term_templates,
    mathy_term_string,
    rand_bool,
    rand_op,
)

parser = ExpressionParser()


def generate_newline_q_a(
    file_base: str,
    number: int,
    exclude: Optional[Set[str]] = None,
    eval: bool = False,
    max_len: int = 128,
):

    train_file = f"{file_base}.txt"
    if exclude is None:
        exclude = set()
    skips = 0
    skip_threshold = 50000
    problems = 0
    half = number // 2
    positives: int = 0
    negatives: int = 0
    with Path(train_file).open("w") as f:
        with tqdm(total=number, mininterval=0.25, desc=file_base) as pbar:
            while problems < number:
                text, answer = make_problem(positives >= half, negatives >= half)
                if text in exclude or len(text) >= max_len:
                    skips += 1
                    if skips >= skip_threshold:
                        raise ValueError(
                            f"Failed to generate more unique problems after {skips} tries!"
                        )
                    continue
                # Make sure the dataset is split 50/50 for labels
                if answer == 2:
                    if positives >= half:
                        continue
                    positives += 1
                elif answer == 0:
                    if negatives >= half:
                        continue
                    negatives += 1

                skips = 0
                exclude.add(text)
                f.write(f"{text}\n{answer}\n")
                pbar.update(1)
                problems += 1


def make_problem(
    skip_positives: bool = False, skip_negatives: bool = False
) -> Tuple[str, int]:
    left_tpl, right_tpl = get_rand_term_templates(2, exponent_probability=0.5)
    left_base = f"{left_tpl.variable}^{left_tpl.exponent}"
    assert left_base != right_tpl
    problem_type = random.randint(0, 3)
    if problem_type < 2 and skip_negatives:
        problem_type = 2
    elif problem_type == 2 and skip_positives:
        problem_type = 0

    if problem_type == 0:
        # Generate a problem with 0 like terms, and a random operator
        operator = random.choice(["+", "-", "*"])
        problem = f"{left_tpl.make()} {operator} {right_tpl.make()}"
        count = 0
    elif problem_type == 1:
        # Generate a problem with like variables but unlike exponents
        operator = random.choice(["+", "-", "*"])
        other_exp = None if left_tpl.exponent is not None else random.randint(1, 12)
        right_tpl = MathyTermTemplate(variable=left_tpl.variable, exponent=other_exp)
        problem = f"{left_tpl.make()} {operator} {right_tpl.make()}"
        count = 0
    elif problem_type == 2 or problem_type == 3:
        # Generate a problem with 2 like terms, and a random operator
        operator = random.choice(["+", "-"])
        problem = f"{left_tpl.make()} {operator} {left_tpl.make()}"
        count = 2
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    return problem, count


def main(
    name: str,
    train_size: int = 100 * 1000,
    eval_size: int = 1000,
    max_len: int = 32,
    include_eval: bool = True,
    include_generalization: bool = True,
):
    current = os.path.dirname(__file__)
    train_file = os.path.join(current, f"{name}.train")
    eval_file = os.path.join(current, f"{name}.eval")
    generalization_file = os.path.join(current, f"{name}.generalization")

    seen_texts: Set[str] = set()

    print(f"Generating train dataset with {train_size} examples...")
    generate_newline_q_a(train_file, train_size, seen_texts, max_len=max_len)
    if include_eval:
        print(f"Generating eval dataset with {eval_size} examples...")
        generate_newline_q_a(eval_file, eval_size, seen_texts, max_len=max_len)

    if include_generalization:
        print(f"Generating generalization dataset with {eval_size} examples...")
        generate_newline_q_a(
            generalization_file, eval_size, seen_texts, max_len=max_len, eval=True,
        )


if __name__ == "__main__":
    typer.run(main)
