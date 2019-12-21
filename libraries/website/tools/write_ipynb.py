from nbformat import v3, v4
import sys
import os


def render_ipynb(from_file: str, to_file: str):
    # print(f"{from_file} -> {to_file}")
    header = """
# This file is generated from a mathy documentation code snippet.
!pip install mathy
"""

    with open(from_file, "r") as fpin:
        text = fpin.read()

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
