import sys
from subprocess import check_output

if len(sys.argv) < 2:
    raise ValueError("Requires a space delimited list of files")

files = list(sys.argv[1:])
args = ["git", "add"]
for input_file in files:
    args.append(input_file.replace(".py", ".ipynb"))
check_output(args)
