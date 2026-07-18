#!/usr/bin/env python
"""
apply_migrations.py — Apply migrations/*.sql to DATABASE_URL, in filename order.

A plain SQL runner, not Alembic: four tables don't justify a migration
framework. Tracks applied migrations in a `schema_migrations` table so re-runs
are idempotent — running this twice is safe.

Usage:
    python scripts/apply_migrations.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATABASE_URL  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


async def main() -> None:
    if not DATABASE_URL:
        print("DATABASE_URL is not set — nothing to do.", file=sys.stderr)
        sys.exit(1)

    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        print(f"No .sql files found in {MIGRATIONS_DIR}")
        return

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await conn.execute(
            "create table if not exists schema_migrations "
            "(filename text primary key, applied_at timestamptz default now())"
        )
        applied_rows = await conn.fetch("select filename from schema_migrations")
        applied = {r["filename"] for r in applied_rows}

        for f in files:
            if f.name in applied:
                print(f"skip  {f.name} (already applied)")
                continue
            sql = f.read_text(encoding="utf-8")
            print(f"apply {f.name} ...")
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "insert into schema_migrations (filename) values ($1)", f.name
                )
            print("  done")
    finally:
        await conn.close()

    print("Migrations up to date.")


if __name__ == "__main__":
    asyncio.run(main())
