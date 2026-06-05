"""Enable ``python -m taxi_etl`` as an alias for the ``taxi-etl`` CLI."""

from __future__ import annotations

import sys

from taxi_etl.cli import main

if __name__ == "__main__":
    sys.exit(main())
