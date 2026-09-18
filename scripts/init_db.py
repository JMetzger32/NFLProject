"""Create the backbone schema. Safe to re-run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import get_engine, init_schema  # noqa: E402
from sqlalchemy import text  # noqa: E402

if __name__ == "__main__":
    init_schema()
    with get_engine().connect() as conn:
        tables = conn.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' ORDER BY table_name"
            )
        ).scalars().all()
    print(f"Schema applied. {len(tables)} tables present:")
    for t in tables:
        print(f"  {t}")
