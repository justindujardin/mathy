### Project Structure

Mathy has several different python projects including [mathy/mathy_core](https://github.com/mathy/mathy_core) and [mathy/mathy_envs](https://github.com/mathy/mathy_envs).

Each project contains a `tools` folder with a set of bash scripts inside for configuring and executing the tasks related to a specific project.

Each project must have at least the following:

- `setup.sh` does any setup such as creating virtual-environments or installing node modules
- `build.sh` builds any output assets such as python `whl` files or bundled javascript applications
- `test.sh` executes any tests such as with `pytest` for Python or `jest` for Javascript

### Setup

From the root folder, run the `tools/setup.sh` script, which will install the pre-requisites:

```bash
sh tools/setup.sh
```

### Use your local version of Mathy

You can install the **Mathy** python package from the file system:

```bash
pip install -e ./mathy
```

Then use the mathy CLI directly:

```bash
mathy simplify "4x + 2y + 6x"
```

### Code

If you know that you want to make a code contribution or make significant changes to the documentation site, there are bash scripts that set up and build the various parts of Mathy.

!!! info "Untested with Windows."

      Mathy uses `bash` scripts to set up and build its various projects. Running on windows may be complicated by this, but the scripts are simple enough to be ported.

      If you are interested in doing this work, [open an issue here](https://github.com/justindujardin/mathy/issues/new?title=Windows Development Build Scripts){target=\_blank} or submit a PR.

### Docs

The documentation uses <a href="https://www.mkdocs.org/" target="_blank">MkDocs</a>.

All the documentation is in Markdown format in the directory `./website/docs`.

Many of the pages include blocks of code that are complete, runnable applications that live in the `./website/docs/snippets/` directory.

#### Docs for tests

All of the inline python snippets double as tests run with each website build. This helps ensure that:

- The documentation is up to date.
- The documentation examples can be run as-is.
- The documentation examples can be launched in [Colab](https://colab.research.google.com/){target=\_blank}.

After executing the test snippets, we convert each file to an [iPython notebook](https://ipython.org/notebook.html){target=\_blank} and an "Open in Colab" badge is inserted near the snippet.

Having a one-click option for running code snippets allows users to launch into examples and run them directly from their browser.

During local development, there is a script that builds the site and checks for any changes, live-reloading:

```bash
cd libraries/website
sh tools/develop.sh
```

It will serve the documentation on `http://127.0.0.1:8000`.

That way, you can edit the documentation/source files and see the changes live.

### Tests

There is a script that you can run locally to test all the code and generate coverage reports in HTML.

From the root folder, run:

```bash
sh tools/test.sh
```

This command generates a directory `./htmlcov/`. If you open the file `./htmlcov/index.html` in your browser, you can interactively explore the regions of code covered by test execution.
