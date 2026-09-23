"""Database engine/session handling.

SQLite is a deliberate choice here: the data volume is small (a run adds ~500 rows),
it needs no server, and the whole QC history is a single file that can be copied or
backed up. Foreign keys are enabled explicitly because SQLite leaves them off by
default, which would otherwise let cascading deletes silently orphan rows.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_config
from src.db.models import Base

_engines: dict[Path, Engine] = {}
_factories: dict[Path, sessionmaker[Session]] = {}


def resolve_db_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path) if db_path else get_config().database_path


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Return (and cache) the engine for a database file, creating its folder."""
    path = resolve_db_path(db_path)
    if path not in _engines:
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{path}", future=True)

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record):  # pragma: no cover
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        _engines[path] = engine
        _factories[path] = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return _engines[path]


def init_db(db_path: str | Path | None = None) -> Path:
    """Create any missing tables. Safe to call on every startup."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return resolve_db_path(db_path)


@contextmanager
def session_scope(db_path: str | Path | None = None) -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on any exception.

    Ingestion relies on this so a run that fails validation part-way through never
    leaves a half-written run in the history.
    """
    get_engine(db_path)
    factory = _factories[resolve_db_path(db_path)]
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Drop cached engines. Used by tests that point at temporary databases."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _factories.clear()
