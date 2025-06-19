# Contributing guidelines

## Before contributing

Welcome to arviz-base! Before contributing to the project,
make sure that you **read our code of conduct** (`CODE_OF_CONDUCT.md`).

## Contributing code

1. Set up a Python development environment
   (advice: use [venv](https://docs.python.org/3/library/venv.html),
   [virtualenv](https://virtualenv.pypa.io/), or [miniconda](https://docs.conda.io/en/latest/miniconda.html))
2. Install tox: `python -m pip install tox`
3. Clone the repository
4. Start a new branch off main: `git switch -c new-branch main`
5. Install dependencies `tox devenv -e py310 .venv`. You may need to install some extra dependencies `pip install ".[check,doc,ci]"`.
   (change the version number according to the Python you are using, and `.venv` for your environment path)
6. Make your code changes
7. Check that your code follows the style guidelines of the project: `tox -e check`
8. (optional) Build the documentation: `tox -e docs`
9.  (optional) Run the tests: `tox -e py310`
   (change the version number according to the Python you are using)
10.  Commit, push, and open a pull request!
