"""Shared pytest fixtures for every test module.

`env` (defined in test_backend) is a Flask app wired to a fake LM Studio and a
temporary data dir and workspace; re-exporting it here makes it available to
all test files without each one importing it.
"""

from test_backend import env  # noqa: F401  (re-exported fixture)
