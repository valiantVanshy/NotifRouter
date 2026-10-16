"""Convenience entry point: `python main.py` routes dataset/messages.csv into dataset/output.csv.

Any arguments are passed to the `route` command, e.g. `python main.py --offline`.
For the full CLI use `python -m notif_router.cli --help`.
"""

import sys

from notif_router.cli import main

if __name__ == "__main__":
    sys.exit(main(["route", *sys.argv[1:]]))
