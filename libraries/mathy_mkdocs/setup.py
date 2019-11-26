import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf8") as f:
    requirements = [line.strip() for line in f]

setuptools.setup(
    name="mathy_mkdocs",
    version="0.0.1dev1",
    author="Justin DuJardin",
    author_email="justin@dujardinconsulting.com",
    description="mkdocs plugin for rendering mathy expressions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justindujardin/mathy",
    packages=setuptools.find_packages(),
    install_requires=["svgwrite"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    entry_points={"mkdocs.plugins": ["mathy = mathy_mkdocs:MathyMkDocsPlugin"]},
)
