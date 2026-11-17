"""Database engine/session setup and Alembic-driven migrations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from . import config

engine = create_engine(config.DATABASE_URL, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA journal_mode=WAL")
    cur.close()


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def alembic_config() -> Config:
    cfg = Config(str(config.ALEMBIC_INI))
    cfg.set_main_option("script_location", str(config.ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", config.DATABASE_URL)
    return cfg


def upgrade_db() -> None:
    """Bring the database to the latest migration (creates it on first run)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    command.upgrade(alembic_config(), "head")
