import os
from pathlib import Path
import re
import sys
from subprocess import check_output


def convert_input(input_location: str):
    check_output(
        ["../.env/bin/jupyter", "nbconvert", "--to", "markdown", input_location]
    )

    # open the output file with pathlib, then call remove_ansi_codes its contents
    # and write it back to the file
    out_path = Path(input_location.replace(".ipynb", ".md"))
    out_text = out_path.read_text()
    out_path.write_text(remove_ansi_codes(out_text))


def remove_ansi_codes(text: str) -> str:
    # Regular expression for matching ANSI escape codes
    ansi_escape = re.compile(r"\x1b\[([0-9A-Za-z;]+)m")
    return ansi_escape.sub("", text)


if len(sys.argv) > 1:
    files = list(sys.argv[1:])
    for input_file in files:
        convert_input(input_file)
else:
    source_path = "docs/examples/"
    for dir_name, _, files in os.walk(source_path):
        rel_path = dir_name.replace(f"{source_path}", "")
        if rel_path != "" and rel_path[0] == "/":
            rel_path = rel_path[1:]
        if "__pycache__" in rel_path or ".ipynb_checkpoints" in rel_path:
            continue
        print(f"Found directory: {rel_path}")
        for file_name in files:
            if os.path.splitext(file_name)[-1] != ".ipynb":
                continue
            print(f"\t{file_name}")
            from_file = os.path.join(source_path, rel_path, file_name)
            convert_input(from_file)
