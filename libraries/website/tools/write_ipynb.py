import os
import sys

from nbformat import v3, v4


def render_ipynb(from_file: str, to_file: str):
    # print(f"{from_file} -> {to_file}")
    header = """
# This file is generated from a Mathy (https://mathy.ai) code example.
!pip install mathy --upgrade
"""

    with open(from_file, "r") as fpin:
        lines = fpin.readlines()
    header_installs = True
    out_lines = []
    for line in lines:
        # NOTE: the weird use of f-string is to workaround vscode highlight
        #       getting really confused by the "!pip install" attached to a #
        if line.startswith(f"#{'!pip install'}"):
            if header_installs is False:
                raise ValueError(
                    "All !pip install comments must be the first lines in a snippet."
                    f" Found the following line after a non-install comment: {line}"
                )
            # output without the comment so ipynb installs the requirement
            out_lines.append(line[1:])
            continue
        # The header installs must be the first (n) lines in a file. After the
        # first non-comment, nothing will be installed.
        header_installs = False
        out_lines.append(line)
    text = "".join(out_lines)
    nbook = v4.upgrade(v3.reads_py(f"{header}{text}"))
    with open(to_file, "w") as fpout:
        fpout.write(f"{v4.writes(nbook)}\n")


if len(sys.argv) > 1:
    files = list(sys.argv[1:])
    for input_file in files:
        output_file = input_file.replace(".py", ".ipynb")
        print(output_file)
        render_ipynb(input_file, output_file)
else:
    source_path = "docs/snippets/"
    for dir_name, _, files in os.walk(source_path):
        rel_path = dir_name.replace(f"{source_path}", "")
        if rel_path != "" and rel_path[0] == "/":
            rel_path = rel_path[1:]
        if "__pycache__" in rel_path:
            continue
        # print(f"Found directory: {rel_path}")
        for file_name in files:
            if file_name in ["__init__.py"] or os.path.splitext(file_name)[-1] != ".py":
                continue
            # print(f"\t{file_name}")
            from_file = os.path.join(source_path, rel_path, file_name)
            out_file = file_name.replace(".py", ".ipynb")
            to_file = os.path.join(source_path, rel_path, out_file)
            print(out_file)
            render_ipynb(from_file, to_file)
