import re
from typing import Dict, List, Tuple


def calculate_term_grouping_distances(input: str) -> Tuple[float, float]:
    vars = re.findall(r"([a-zA-Z]\^?\d?)+", input)
    seen_pos: Dict[str, List[int]] = dict()
    for i, var in enumerate(vars):
        if var not in seen_pos:
            seen_pos[var] = []
        seen_pos[var].append(i)

    def get_var_signal(var, positions):
        out_signal = 0.0
        last_pos = -1
        for position in positions:
            if last_pos != -1:
                out_signal += position - last_pos
            last_pos = position
        return out_signal

    # seen_pos is a dict of positions that each variable is seen at
    # add up all the distances between the points in each variable
    signal = 0.0
    for key, value in seen_pos.items():
        signal += get_var_signal(key, value)

    # Scale the signal down to avoid it growing ever larger with more
    # complex inputs.
    return signal, signal / len(vars)


def calculate_grouping_control_signal(
    input: str, output: str, clip_at_zero: bool = False
) -> float:
    """Calculate grouping_control signals as the sum of all distances between
    all like terms. Gather all the terms in an expression and add an error value
    whenever a like term is separated by another term.

    Examples:
        "2x + 2x" = 0
        "2x + 4y + 2x" = 1
        "2x + 4y + 2x + 4y" = 2
        "2x + 2x + 4y + 4y" = 0
    """

    # We cheat the term grouping a bit by not parsing the expression
    # and finding the real set of terms. Instead we remove all the non-variables
    # and then count the distances from the resulting string.
    #
    # NOTE: this means that the signal is not correct when exponents or complex
    #       terms with multiple variables are in the expression. Perhaps it's a
    #       good improvement to make here.
    in_signal, in_signal_normalized = calculate_term_grouping_distances(input)
    out_signal, out_signal_normalized = calculate_term_grouping_distances(output)
    # The grouping distance stayed the same
    if in_signal == out_signal:
        return out_signal_normalized

    # It changed, no error
    if clip_at_zero is True:
        return 0.0

    # It changed, negative error based on magnitude of the change
    return -abs(in_signal_normalized - out_signal_normalized)
