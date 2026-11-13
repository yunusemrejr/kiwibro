"""Allow `python3 -m {{name}}` to run the tool."""

import sys

from {{name}}.cli import main

if __name__ == "__main__":
    sys.exit(main())
