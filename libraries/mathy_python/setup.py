from setuptools import setup, find_packages

# DO_NOT_MODIFY_THIS_VALUE_IS_SET_BY_THE_BUILD_MACHINE
VERSION = "1.3.0"


def readme():
    with open("README.md") as f:
        return f.read()


with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as file:
    REQUIRED_MODULES = [line.strip() for line in file]

with open("requirements-dev.txt") as file:
    DEVELOPMENT_MODULES = [line.strip() for line in file]


setup(
    name="mathy",
    version=VERSION,
    description="making math fantastic",
    long_description=readme(),
    keywords="math",
    url="https://github.com/justindujardin/mathy",
    author="Justin DuJardin",
    author_email="justin@dujardinconsulting.com",
    packages=find_packages(),
    install_requires=REQUIRED_MODULES,
    extras_require={"dev": DEVELOPMENT_MODULES},
    package_data={"mathy": ["tests/api/*.json", "tests/rules/*.json"]},
    entry_points="""
        [console_scripts]
        mathy=mathy.cli:cli
    """,
    include_package_data=True,
)
