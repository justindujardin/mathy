from pathlib import Path
from subprocess import check_output
from ruamel.yaml import YAML

# Constants
EXCLUDE_FILES = {
    ".DS_Store",
    "__init__.py",
    "README.md",
    "types.py",
    "py.typed",
    "about.py",
    "conrastive.py",
}
INCLUDE_FOLDERS = ["."]
DOCS_KEY = "API"

# Paths
package_name = "mathy"
parent_folder_path = Path(__file__).parent.parent
yaml_path = parent_folder_path / "mkdocs.yml"
source_path = parent_folder_path.parent / package_name
dest_dir = parent_folder_path / "docs" / "api"

# YAML
yaml = YAML()


def prepend_md_content(original_md, prepending_md):
    with open(prepending_md, "r") as file:
        prepending_content = file.read()

    with open(original_md, "r+") as file:
        original_content = file.readlines()
        # First line is a heading 1 with the full module name that is described.
        # Make it explode if that changes so we notice.
        assert original_content[0].startswith(
            "# "
        ), "Expected heading 1 at beginning of mathy_pydoc API doc file"

        # Change to
        ticks = "```"
        import_statement = f"{ticks}python\n\nimport {original_content[0][2:]}{ticks}"
        new_content = f"## API\n\n"
        module_namespace = import_statement
        original_content[0] = new_content
        # flatten the list of lines into a single string
        original_content = "".join(original_content)

        file.seek(0, 0)
        file.write(f"{module_namespace}\n{prepending_content}\n\n{original_content}")


def h1_to_h2(original_md: str):
    file_path = Path(original_md)
    original_content = file_path.read_text().splitlines()
    # First line is a heading 1 with the full module name that is described.
    # Make it explode if that changes so we notice.
    assert original_content[0].startswith(
        "# "
    ), "Expected heading 1 at beginning of mathy_pydoc API doc file"
    ticks = "```"
    original_content[0] = f"{ticks}python\n\nimport {original_content[0][2:]}\n{ticks}"
    # flatten the list of lines into a single string
    original_content = "\n".join(original_content)
    file_path.unlink()
    file_path.write_text(original_content)


def render_docs(src_rel_path, src_file, to_file, modifier="++"):
    insert = "." + src_rel_path if src_rel_path not in ["", "."] else ""
    namespace = f"{package_name}{insert}.{src_file.stem}{modifier}"
    args = ["mathy_pydoc", "--plain", namespace]
    if not to_file.parent.exists():
        to_file.parent.mkdir(parents=True)
    call_result = check_output(args, cwd=parent_folder_path).decode("utf-8")
    with open(to_file, "w") as file:
        file.write(call_result)


def process_directory(directory):
    nav_entries = []
    for file_path in directory.iterdir():
        if file_path.name in EXCLUDE_FILES or not file_path.suffix == ".py":
            continue

        print(f"\t{file_path.name}")
        rel_out_md = file_path.with_suffix(".md").name
        to_file = dest_dir / directory.relative_to(source_path) / rel_out_md
        render_docs(directory.relative_to(source_path).as_posix(), file_path, to_file)

        # Prepend existing md file content if present
        existing_md = file_path.with_suffix(".md")
        if existing_md.exists():
            prepend_md_content(to_file, existing_md)
        else:
            h1_to_h2(to_file)

        nav_item = {
            file_path.stem: to_file.relative_to(parent_folder_path / "docs").as_posix()
        }
        nav_entries.append(nav_item)
    return nav_entries


def update_yaml_nav(nav_entries):
    mkdocs_yaml = yaml.load(yaml_path)
    updated = False
    site_nav = mkdocs_yaml["nav"]
    for nav_obj in site_nav:
        site_keys = list(nav_obj.keys())
        for key in site_keys:
            if isinstance(nav_obj[key], str):
                continue
            for nav_sub in nav_obj[key]:
                if DOCS_KEY in nav_sub:
                    nav_sub[DOCS_KEY] = nav_entries
                    updated = True
                    break
    if not updated:
        raise Exception(f"Could not find {DOCS_KEY} in mkdocs.yml")
    with open(yaml_path, "w") as file:
        yaml.dump(mkdocs_yaml, file)


def main():
    print("Building API docs...")
    nav_entries = []
    for src_folder in INCLUDE_FOLDERS:
        folder = source_path / src_folder
        if folder.is_dir():
            print(f"Found directory: {folder.relative_to(source_path)}")
            if src_folder not in [".", ""]:
                new_entries = process_directory(folder)
                new_entries.sort(key=lambda x: list(x)[0])
                nav_entries.append({folder.name: new_entries})
            else:
                nav_entries += process_directory(folder)
    nav_entries.sort(key=lambda x: list(x)[0])
    update_yaml_nav(nav_entries)
    print("Done!")


if __name__ == "__main__":
    main()
