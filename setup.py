import setuptools
import pathlib

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f]

setuptools.setup(
    name="mathy",
    version="0.0.1dev1",
    author="Justin DuJardin",
    author_email="justin@dujardinconsulting.com",
    description="RL agent and CAS library that solve math problems step-by-step",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justindujardin/mathy",
    packages=setuptools.find_packages(),
    package_data={"mathy": ["agents/*"]},
    install_requires=requirements,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
