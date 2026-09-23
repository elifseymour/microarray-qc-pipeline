"""Load every run found in a folder of `*_densities.csv` / `*_run_metadata.csv` pairs.

The dashboard's upload page is the normal way in; this exists to seed or rebuild the
database quickly during development and after regenerating the synthetic dataset.

    python -m src.ingestion.bulk_load --fresh
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PROJECT_ROOT, get_config
from src.db.session import init_db, reset_engine_cache
from src.ingestion.ingest_run import IngestionError, ingest_run

DENSITY_SUFFIX = "_densities.csv"
METADATA_SUFFIX = "_run_metadata.csv"


def discover_run_files(folder: Path) -> list[tuple[str, Path, Path]]:
    """Find (run_id, density file, metadata file) triples, ordered by run id."""
    triples = []
    for density in sorted(folder.glob(f"*{DENSITY_SUFFIX}")):
        run_id = density.name[: -len(DENSITY_SUFFIX)]
        metadata = folder / f"{run_id}{METADATA_SUFFIX}"
        if metadata.exists():
            triples.append((run_id, density, metadata))
    return triples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--folder", type=Path, default=PROJECT_ROOT / "data" / "raw", help="folder to scan"
    )
    parser.add_argument("--db", type=Path, default=None, help="target database file")
    parser.add_argument(
        "--fresh", action="store_true", help="delete the database first, then load"
    )
    parser.add_argument(
        "--replace", action="store_true", help="overwrite runs that are already ingested"
    )
    args = parser.parse_args()

    db_path = args.db or get_config().database_path
    if args.fresh and Path(db_path).exists():
        reset_engine_cache()
        Path(db_path).unlink()
        print(f"Removed existing database {db_path}")
    init_db(db_path)

    triples = discover_run_files(args.folder)
    if not triples:
        raise SystemExit(f"No run file pairs found in {args.folder}")

    loaded = skipped = 0
    for run_id, density, metadata in triples:
        try:
            result = ingest_run(
                density,
                metadata,
                db_path=db_path,
                replace=args.replace or args.fresh,
                density_name=density.name,
                metadata_name=metadata.name,
            )
        except IngestionError as exc:
            skipped += 1
            print(f"  SKIP {run_id}: {exc}")
            continue
        loaded += 1
        print(
            f"  {result.spotting_run_id}  {result.panel}  {result.run_date}  "
            f"{result.n_chips} chips, {result.n_chips_failed} failed "
            f"({result.failure_rate_pct:.1f}%)"
        )
        for warning in result.warnings:
            print(f"      note: {warning}")

    print(f"\nLoaded {loaded} run(s), skipped {skipped}. Database: {db_path}")


if __name__ == "__main__":
    main()
