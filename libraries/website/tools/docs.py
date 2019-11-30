import json

# Import the os module, for the os.walk function
import os
import tempfile
from pathlib import Path
from subprocess import check_output
from typing import Dict, List, Optional, Any

from ruamel.yaml import YAML

parent_folder_path = Path(__file__).parent.parent

trash_folder = str(parent_folder_path / ".trash/")
if not os.path.exists(trash_folder):
    os.mkdir(trash_folder)


yaml_path = parent_folder_path / "mkdocs.yml"
source_path = parent_folder_path.parent / "mathy_python" / "mathy"

dest_dir = str(parent_folder_path / "docs" / "api")
yaml = YAML()


print("Building API docs...")
exclude_files = ["__init__.py", "README.md"]
include_folders = ["core"]


def render_docs(src_rel_path: str, src_file: str, to_file: str, modifier="++"):
    src_rel_ns = src_rel_path.replace("/", ".")
    src_base = src_file.replace(".py", "")
    # pydocmd simple mathy.core.expressions++ > ../website/docs/api/core/expressions.md
    file_dir = os.path.dirname(__file__)
    # Set CWD to the root
    os.chdir(os.path.join(file_dir, "../"))

    args = [
        "pydocmd",
        "simple",
        f"mathy.{src_rel_ns}.{src_base}{modifier}",
    ]
    # print(args)
    call_result = check_output(args, env=os.environ).decode("utf-8")
    with open(to_file, "w") as file:
        file.write(call_result)

    # shutil.rmtree(dest_dir)
    print(f"Done, output: {dest_dir}")


if __name__ == "__main__":
    print(f"Render to: {dest_dir}")

    nav_entries = []
    for dir_name, _, files in os.walk(source_path):
        rel_path = dir_name.replace(f"{source_path}/", "")
        if "__pycache__" in rel_path or "/lib" in rel_path or "trfl" in rel_path:
            continue
        if rel_path not in include_folders:
            continue
        print(f"Found directory: {rel_path}")
        target_dir = os.path.join(dest_dir, rel_path)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        found = []
        for file_name in files:
            if file_name in exclude_files:
                continue
            print(f"\t{file_name}")
            rel_out_base = file_name.replace(".py", "")
            rel_out_md = f"{rel_out_base}.md"
            to_file = os.path.join(dest_dir, rel_path, rel_out_md)
            nav_item: Any = dict()
            nav_item[os.path.basename(rel_out_base)] = os.path.join(
                "api", rel_path, rel_out_md
            )
            found.append(nav_item)
            render_docs(
                rel_path, file_name, to_file,
            )
        # Only add to nav if any files were found
        if len(found) > 0:
            nav_key = os.path.basename(rel_path)
            nav_item = dict()
            nav_item[nav_key] = found
            nav_entries.append(nav_item)

    # pydocmd simple mathy.core.expressions++ > ../website/docs/api/core/expressions.md
    YAMLSection = List[Dict[str, List[Dict[str, str]]]]

    mkdocs_yaml = yaml.load(yaml_path)
    docs_key = "API Documentation"
    site_nav = mkdocs_yaml["nav"]
    for nav_obj in site_nav:
        if docs_key in nav_obj:
            nav_obj[docs_key] = nav_entries
            break

    out = mkdocs_yaml
    with open(yaml_path, "w") as file:
        yaml.dump(mkdocs_yaml, file)
    print("done!")
