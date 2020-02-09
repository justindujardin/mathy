# Copyright (C) 2016  Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import setuptools
import io

from pathlib import Path

package_name = "mathy_pydoc"
root = Path(__file__).parent.resolve()

# Read in package meta from about.py
about_path = root / package_name / "about.py"
with about_path.open("r", encoding="utf8") as f:
    about = {}
    exec(f.read(), about)


with io.open("README.md", encoding="utf8") as fp:
    readme = fp.read()

setuptools.setup(
    name=package_name,
    description=about["__summary__"],
    author=about["__author__"],
    author_email=about["__email__"],
    url=about["__uri__"],
    version=about["__version__"],
    license=about["__license__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="markdown pydoc generator docs documentation",
    packages=["mathy_pydoc"],
    install_requires=["Markdown>=2.6.11", "PyYAML>=3.12", "six>=0.11.0",],
    entry_points=dict(console_scripts=["mathy_pydoc=mathy_pydoc.__main__:main"]),
)
