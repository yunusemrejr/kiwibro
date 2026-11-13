# Tests for {{module}}

A starting point for pytest tests around an existing module. The kit expects
`{{module}}.py` to provide:

- `add(a, b)`: the sum of two numbers
- `parse_int(text)`: an int, raising `ValueError("not an integer")` on bad input
- `count_lines(path)`: the number of lines in a text file

Point the imports and cases at the module's real functions, then run
`python3 -m pytest -q`. Shared fixtures live in conftest.py; the cases show
parametrize for several inputs, pytest.raises for errors and tmp_path for files.
