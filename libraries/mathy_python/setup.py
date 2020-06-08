from pathlib import Path
from setuptools import setup, find_packages


def setup_package():
    # from mathy.about import PackageExtras

    package_name = "mathy"
    root = Path(__file__).parent.resolve()

    # Read in package meta from about.py
    about_path = root / package_name / "about.py"
    with about_path.open("r", encoding="utf8") as f:
        about = {}
        exec(f.read(), about)
        PackageExtras = about["PackageExtras"]

    with open(root / "README.md", "r") as fh:
        long_description = fh.read()

    with open(root / "requirements.txt") as file:
        REQUIRED_MODULES = [line.strip() for line in file]

    with open(root / "requirements-dev.txt") as file:
        DEVELOPMENT_MODULES = [line.strip() for line in file if "-e" not in line]

    extras = {
        "dev": DEVELOPMENT_MODULES,
        PackageExtras.AGENTS: [
            "tensorflow>=2.1.0,<2.2.0",
            "tensorboard>=2.1.0,<2.2.0",
            "tensorflow_probability",
            "keras-self-attention",
            "gym",
        ],
        PackageExtras.SOLVER: ["fragile==0.0.47", "tqdm>=4.43.0", "gym"],
        PackageExtras.REFORMER: ["torch", "reformer_pytorch", "thinc>=8.0.0a6"],
    }
    extras["all"] = [item for group in extras.values() for item in group]

    setup(
        name=package_name,
        description=about["__summary__"],
        author=about["__author__"],
        author_email=about["__email__"],
        url=about["__uri__"],
        version=about["__version__"],
        license=about["__license__"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords="math",
        install_requires=REQUIRED_MODULES,
        packages=find_packages(),
        extras_require=extras,
        package_data={"mathy": ["tests/api/*.json", "tests/rules/*.json"]},
        entry_points="""
            [console_scripts]
            mathy=mathy.cli:cli
        """,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Development Status :: 2 - Pre-Alpha",
        ],
        python_requires=">=3.6",
        include_package_data=True,
    )


if __name__ == "__main__":
    setup_package()
