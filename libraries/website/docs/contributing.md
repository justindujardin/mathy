First, you might want to see the basic ways to [help with Mathy and get help](/help){target=\_blank}.

### Project Structure

Mathy has a number of different python projects inside the `libraries` folder.

Each project contains a `tools` folder that has a set of bash scripts inside for configuring and executing the tasks related to a specific project.

Each project is expected to have at least the following:

- `setup.sh` does any setup such as creating virutal environments, or installing node modules
- `build.sh` builds any output assets such as python `whl` files or bundled javascript applications
- `test.sh` execute any tests such as with `pytest` for Python or `jest` for Javascript

### Setup

Mathy can either build everything or specific sub-projects.

#### All Projects

From the root folder, run the `tools/setup.sh` script, which will install the pre-requisites for all projects in the `libraries` folder:

```bash
sh tools/setup.sh
```

#### One Project

From the root folder, change directories into the desired project folder, e.g.

```bash
cd libraries/website
```

From the project folder, run the setup script:

```bash
sh tools/setup.sh
```

### Use your local version of Mathy

You can install the **Mathy** python package from the file system:

```bash
pip install -e ./libraries/mathy_python
```

Then use the mathy CLI directly:

```bash
mathy simplify "4x + 2y + 6x"
```

### Code

If you know that you want to make a code contribution, or you want to make major changes to the documentation site, there are a set of bash scripts that setup and build the various parts of Mathy.

!!! info "Untested with Windows"

      Mathy uses `bash` scripts to setup and build its various projects. Running on windows may be complicated by this, but the scripts are simple enough that they could be ported.

      If you are interested in doing this work, [open an issue here](https://github.com/justindujardin/mathy/issues/new?title=Windows Development Build Scripts){target=\_blank} or submit a PR.

### Docs

The documentation uses <a href="https://www.mkdocs.org/" target="_blank">MkDocs</a>.

All the documentation is in Markdown format in the directory `./libraries/website/docs`.

Many of the pages include blocks of code.

In most of the cases, these blocks of code are actual complete applications that can be run as is.

In fact, those blocks of code are not written inside the Markdown, they are Python files in the `./libraries/website/docs/snippets/` directory.

And those Python files are included/injected in the documentation when generating the site.

#### Docs for tests

All of the inline python snippets act as tests that are run during each site build. If a snippet fails to
execute, the build fails.

This helps making sure that:

- The documentation is up to date.
- The documentation examples can be run as is.
- The documentation examples can be launched in [Colab](https://colab.research.google.com/){target=\_blank}.
- Most of the features are covered by the documentation, ensured by the coverage tests.

After executing the test snippets, each file is converted to an [iPython notebook](https://ipython.org/notebook.html){target=\_blank} and an "Open in Colab" badge is inserted by the snippet.

This allows users to launch into examples and run them directly from their browser, without installing any software locally.

During local development, there is a script that builds the site and checks for any changes, live-reloading:

```bash
cd libraries/website
sh tools/develop.sh
```

It will serve the documentation on `http://127.0.0.1:8000`.

That way, you can edit the documentation/source files and see the changes live.

### Tests

There is a script that you can run locally to test all the code and generate coverage reports in HTML.

From the root folder run:

```bash
sh tools/test.sh
```

This command generates a directory `./htmlcov/`, if you open the file `./htmlcov/index.html` in your browser, you can explore interactively the regions of code that are covered by the tests, and notice if there is any region missing.
