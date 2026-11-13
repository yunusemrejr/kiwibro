# {{name}}

A small Python package on the src layout. One function, `slugify(text,
max_length=60)`, turns a title into a URL slug.

## Run the tests

    python3 -m pytest -q

`pythonpath = ["src"]` in pyproject.toml lets pytest import the package without
installing it. To install it, run `python3 -m pip install -e .`. The package
needs nothing outside the standard library.
