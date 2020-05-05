import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np
import typer
from tqdm.auto import tqdm

from mathy import ExpressionParser, MathExpression, TermEx, get_term_ex, get_terms
from mathy.problems import gen_simplify_multiple_terms, mathy_term_string, use_pretty_numbers

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
    skip_threshold = 100
    problems = 0
    min_like = 4 if eval else 2
    max_like = 12 if eval else 8
    with Path(train_file).open("w") as f:
        with tqdm(total=number, mininterval=0.25, desc=file_base) as pbar:
            while problems < number:
                text, complexity = gen_simplify_multiple_terms(
                    random.randint(min_like, max_like),
                    noise_probability=0.5,
                    noise_terms=random.randint(1, 5),
                    op=["+", "-"],
                )

                if text in exclude or len(text) >= max_len:
                    skips += 1
                    if skips >= skip_threshold:
                        raise ValueError(
                            f"Failed to generate more unique problems after {skips} tries!"
                        )
                    continue

                skips = 0
                exclude.add(text)
                answer = count_like_terms(text)
                f.write(f"{text}\n{answer}\n")
                pbar.update(1)
                problems += 1


def count_like_terms(input_problem: str) -> int:
    expression: MathExpression = parser.parse(input_problem)
    term_nodes: List[MathExpression] = get_terms(expression)
    node_groups: Dict[str, List[MathExpression]] = {}
    for term_node in term_nodes:
        ex: Optional[TermEx] = get_term_ex(term_node)
        assert ex is not None, f"invalid expression {term_node}"
        key = mathy_term_string(variable=ex.variable, exponent=ex.exponent)
        if key == "":
            key = "const"
        if key not in node_groups:
            node_groups[key] = [term_node]
        else:
            node_groups[key].append(term_node)
    like_terms = 0
    for k, v in node_groups.items():
        if len(v) <= 1:
            continue
        like_terms += len(v)
    return like_terms


def main(
    name: str,
    train_size: int = 200 * 1000,
    eval_size: int = 1000,
    max_len: int = 128,
    include_eval: bool = True,
    include_generalization: bool = True,
    pretty:bool = False
):

    use_pretty_numbers(pretty)

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
