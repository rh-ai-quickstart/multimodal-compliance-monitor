"""Snapshot the running database into db_seed_data.sql for eval use.

Connects to the live Postgres instance and dumps the 4 eval-relevant tables
(app_config, detection_classes, detection_tracks, detection_observations)
as INSERT statements into the single root db_seed_data.sql file.

Usage::

    python init_db.py                 # writes to ./db_seed_data.sql
    python init_db.py /custom/path    # writes to a custom path

Typically invoked via ``make init-eval-db``.
"""

import sys
from pathlib import Path

from load_seed import _dump_tables_to_file

OUTPUT_PATH = Path(__file__).parent / "db_seed_data.sql"


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_PATH

    print(f"Dumping database to {output} ... ", end="", flush=True)
    count = _dump_tables_to_file(output)
    print(f"done ({count} statements)")
    print(f"Eval seed saved: {output}")


if __name__ == "__main__":
    main()
