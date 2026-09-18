"""PostgreSQL connection + the generic dataframe loader every ingest module uses."""

import io
import re
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import DATABASE_URL, SCHEMA_PATH

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, future=True)
    return _engine


def init_schema() -> None:
    """Apply src/schema.sql. Idempotent."""
    sql = SCHEMA_PATH.read_text()
    with get_engine().begin() as conn:
        conn.execute(text(sql))


def table_exists(table: str) -> bool:
    with get_engine().connect() as conn:
        return bool(
            conn.execute(
                text("SELECT to_regclass(:t) IS NOT NULL"), {"t": f"public.{table}"}
            ).scalar()
        )


def existing_columns(table: str) -> set[str]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = :t"
            ),
            {"t": table},
        )
        return {r[0] for r in rows}


def row_count(table: str) -> int:
    if not table_exists(table):
        return 0
    with get_engine().connect() as conn:
        return int(conn.execute(text(f'SELECT count(*) FROM "{table}"')).scalar())


def to_pandas(df) -> pd.DataFrame:
    """nflreadpy returns polars frames; everything downstream here is pandas."""
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _clean_column(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).lower().strip("_")
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"c_{name}"
    return name[:63]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_column(c) for c in df.columns]
    # Duplicate names can appear after cleaning; keep the first.
    df = df.loc[:, ~df.columns.duplicated()]
    # pandas nullable / arrow-backed dtypes confuse the CSV COPY path.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype("object")
        elif str(df[col].dtype).startswith(("Int", "Float", "boolean", "string")):
            df[col] = df[col].astype("object").where(df[col].notna(), None)
    return df


def _copy_into(table: str, df: pd.DataFrame) -> None:
    """Bulk insert via COPY. Far faster than to_sql for wide play-by-play frames."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="")
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    raw = get_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.copy_expert(
                f'COPY "{table}" ({cols}) FROM STDIN WITH (FORMAT CSV, NULL \'\')',
                buf,
            )
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def _ensure_table(table: str, df: pd.DataFrame) -> None:
    if not table_exists(table):
        df.head(0).to_sql(table, get_engine(), if_exists="fail", index=False)
        return
    missing = [c for c in df.columns if c not in existing_columns(table)]
    if missing:
        # nflverse added columns since the last load; widen rather than drop them.
        ddl = df[missing].head(0)
        tmp = f"_tmp_widen_{table}"[:63]
        ddl.to_sql(tmp, get_engine(), if_exists="replace", index=False)
        with get_engine().begin() as conn:
            types = conn.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema='public' AND table_name=:t"
                ),
                {"t": tmp},
            ).all()
            for col, dtype in types:
                conn.execute(
                    text(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{col}" {dtype}')
                )
            conn.execute(text(f'DROP TABLE IF EXISTS "{tmp}"'))


def _ensure_indexes(
    table: str, key_cols: Sequence[str], index_cols: Iterable[Sequence[str]]
) -> None:
    # Index specs are declared optimistically; a feed may not ship every column
    # (season summaries drop `team`, for instance), so skip groups that don't apply.
    present = existing_columns(table)
    with get_engine().begin() as conn:
        if key_cols:
            cols = ", ".join(f'"{c}"' for c in key_cols)
            name = f"{table}_pkey_uq"[:63]
            conn.execute(
                text(
                    f'CREATE UNIQUE INDEX IF NOT EXISTS "{name}" ON "{table}" ({cols})'
                )
            )
        for group in index_cols:
            if not all(c in present for c in group):
                continue
            cols = ", ".join(f'"{c}"' for c in group)
            name = f"idx_{table}_{'_'.join(group)}"[:63]
            conn.execute(
                text(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ({cols})')
            )


def load_frame(
    df: pd.DataFrame,
    table: str,
    key_cols: Sequence[str] = (),
    season_col: str | None = "season",
    index_cols: Iterable[Sequence[str]] = (),
    source_fn: str = "",
) -> int:
    """Create-or-widen `table` from `df`, replace the seasons it covers, and load it.

    Re-running for the same seasons is safe: existing rows for those seasons are
    deleted first, so a backfill can be repeated without duplicating data.
    """
    if df is None or df.empty:
        _log(table, [], 0, source_fn, "empty", "source returned no rows")
        return 0

    df = _prepare(df)
    key_cols = [_clean_column(c) for c in key_cols]
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        # Silently dropping these would build a unique index on a partial key and
        # reject legitimate rows. Surface the source schema drift instead.
        raise ValueError(
            f"{table}: declared key columns missing from source frame: {missing}"
        )
    season_col = _clean_column(season_col) if season_col else None
    if season_col not in df.columns:
        season_col = None

    if key_cols:
        df = df.dropna(subset=key_cols).drop_duplicates(subset=key_cols, keep="last")

    seasons: list[int] = []
    if season_col:
        seasons = sorted({int(s) for s in df[season_col].dropna().unique()})

    _ensure_table(table, df)
    # The table's types were fixed by whichever frame created it. A later frame can
    # carry the same column as float purely because it spans seasons and picked up
    # NaNs, which COPY would reject for an integer column.
    df = _coerce_to_target(df, column_types(table))

    with get_engine().begin() as conn:
        if seasons:
            conn.execute(
                text(f'DELETE FROM "{table}" WHERE "{season_col}" = ANY(:s)'),
                {"s": seasons},
            )
        else:
            conn.execute(text(f'TRUNCATE TABLE "{table}"'))

    _copy_into(table, df)
    _ensure_indexes(table, key_cols, index_cols)
    _log(table, seasons, len(df), source_fn, "ok", "")
    return len(df)


def column_types(table: str) -> dict[str, str]:
    with get_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema='public' AND table_name=:t"
            ),
            {"t": table},
        ).all()
    return {r[0]: r[1] for r in rows}


def _coerce_to_target(df: pd.DataFrame, types: dict[str, str]) -> pd.DataFrame:
    """Match the declared Postgres types so COPY's input parser accepts every value.

    Mostly this is about integers: a pandas float column writes 12.0, which
    Postgres rejects for an integer column.
    """
    df = df.copy()
    for col in df.columns:
        pg = types.get(col, "text")
        s = df[col]
        if pg in ("integer", "bigint", "smallint"):
            s = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        elif pg in ("numeric", "double precision", "real"):
            s = pd.to_numeric(s, errors="coerce")
        elif pg == "boolean":
            s = s.map(
                lambda v: None if pd.isna(v) else bool(v) if not isinstance(v, str)
                else v.strip().lower() in ("true", "t", "1", "yes")
            )
        elif pg == "date":
            s = pd.to_datetime(s, errors="coerce").dt.date
        elif pg.startswith("timestamp"):
            s = pd.to_datetime(s, errors="coerce", utc=pg.endswith("time zone"))
        df[col] = s.astype("object").where(s.notna(), None)
    return df


def upsert_frame(
    df: pd.DataFrame,
    table: str,
    key_cols: Sequence[str],
    source_fn: str = "",
) -> int:
    """Upsert into an explicitly-declared backbone table (teams/games/betting_lines).

    Only columns that already exist on the target table are written, so extra
    nflverse columns are ignored rather than widening a hand-designed table.
    """
    if df is None or df.empty:
        return 0

    df = _prepare(df)
    types = column_types(table)
    cols = [c for c in df.columns if c in types]
    df = df[cols].dropna(subset=list(key_cols)).drop_duplicates(
        subset=list(key_cols), keep="last"
    )
    df = _coerce_to_target(df, types)

    staging = f"_stg_{table}"[:63]
    col_list = ", ".join(f'"{c}"' for c in cols)
    with get_engine().begin() as conn:
        conn.execute(text(f'DROP TABLE IF EXISTS "{staging}"'))
        # Inheriting the target's exact column types avoids any cast on insert.
        conn.execute(
            text(
                f'CREATE TABLE "{staging}" AS '
                f'SELECT {col_list} FROM "{table}" WITH NO DATA'
            )
        )
    _copy_into(staging, df)

    col_sql = col_list
    updates = [c for c in cols if c not in key_cols]
    set_sql = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in updates)
    conflict = ", ".join(f'"{c}"' for c in key_cols)
    action = f"DO UPDATE SET {set_sql}" if updates else "DO NOTHING"

    with get_engine().begin() as conn:
        conn.execute(
            text(
                f'INSERT INTO "{table}" ({col_sql}) '
                f'SELECT {col_sql} FROM "{staging}" '
                f"ON CONFLICT ({conflict}) {action}"
            )
        )
        conn.execute(text(f'DROP TABLE IF EXISTS "{staging}"'))

    _log(table, [], len(df), source_fn, "ok", "")
    return len(df)


def _log(
    table: str,
    seasons: Sequence[int],
    rows: int,
    source_fn: str,
    status: str,
    message: str,
) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO ingest_log "
                    "(table_name, seasons, row_count, source_fn, finished_at, status, message) "
                    "VALUES (:t, :s, :r, :f, now(), :st, :m)"
                ),
                {
                    "t": table,
                    "s": list(seasons),
                    "r": rows,
                    "f": source_fn,
                    "st": status,
                    "m": message[:2000],
                },
            )
    except Exception:
        pass  # logging must never break a load
