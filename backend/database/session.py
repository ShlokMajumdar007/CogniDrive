from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Any
import logging
import threading

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, scoped_session, Session

from backend.app.config import get_settings
from backend.database.base import Base

# Setup logger for the session manager
logger = logging.getLogger("CogniDrive.DatabaseSession")

# Thread lock to protect initialization operations in multi-threaded runtime
_db_lock = threading.Lock()

# Load global configuration
settings = get_settings()

# Extract local path from SQLite URL and ensure folder creation
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite:///"):
    # Strip protocol prefix to get filesystem path
    db_path_str = db_url[len("sqlite:///") :]
    # If using absolute path on Windows (e.g. sqlite:///c:\...)
    if db_path_str.startswith("/") or ":" in db_path_str:
        # Normalize relative to project root or handle absolute path
        db_path = Path(db_path_str.lstrip("/")).absolute()
    else:
        db_path = Path(db_path_str).absolute()

    # Create target sqlite directory if missing
    db_dir = db_path.parent
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Verified database directory exists: {db_dir}")
    except Exception as e:
        logger.error(f"Failed to create database directory {db_dir}: {e}")

# Build SQLAlchemy SQLite engine with check_same_thread disabled for multi-threading
engine = create_engine(
    db_url,
    connect_args={
        "check_same_thread": False,  # Allows multiple threads (Camera, FastAPI, ML) to share connection
        "timeout": 30,               # Database timeout in seconds for locked states
    },
    pool_pre_ping=True,              # Auto-verifies connection viability before requests
    future=True,                     # Enforces SQLAlchemy 2.0 behaviors
)

# Establish connection performance pragmas on SQLite connections
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """Sets SQLite optimization pragmas for high-throughput edge AI requirements.

    Enables WAL mode, enforces foreign keys, optimizes write syncs, and shifts temp storage
    to RAM to improve thread concurrency and database efficiency.
    """
    cursor = dbapi_connection.cursor()
    try:
        # WAL mode permits readers and writers to operate concurrently without locking the DB
        cursor.execute("PRAGMA journal_mode=WAL;")
        # Enable relational constraint checks
        cursor.execute("PRAGMA foreign_keys=ON;")
        # NORMAL synchronizes writes at critical checkpoints only (safe in WAL mode)
        cursor.execute("PRAGMA synchronous=NORMAL;")
        # Store temp tables/indexes in RAM instead of disk to prevent SSD wear
        cursor.execute("PRAGMA temp_store=MEMORY;")
        # Set database memory cache size to ~64MB (negative value represents kilobytes)
        cursor.execute("PRAGMA cache_size=-64000;")
    except Exception as e:
        logger.error(f"Failed to set SQLite database pragmas: {e}")
    finally:
        cursor.close()


# Core Session Maker Setup
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)

# Scoped session thread-local factory
ScopedSession = scoped_session(SessionLocal)


def create_database() -> None:
    """Initialises the SQLite database and runs DDL table creations.

    Idempotent — if tables or indexes already exist (e.g. on server restart)
    the function logs a debug message rather than raising an exception, because
    an existing DB with all schema objects is a valid and expected state.

    This function is wrapped in a thread lock to ensure safe schema instantiation
    across multiple concurrent worker threads during startup.
    """
    import sqlite3

    with _db_lock:
        logger.info("Initializing database setup and checking schemas.")
        try:
            import backend.database.models  # noqa: F401 — register ORM tables
            create_all_tables()
            logger.info("Database schemas initialized successfully.")
        except Exception as e:
            err_str = str(e).lower()
            if "already exists" in err_str:
                logger.debug(
                    "Database schema already present (restart scenario) — skipping DDL: %s", e
                )
            else:
                logger.error("Critical error creating database: %s", e)
                raise



def create_all_tables() -> None:
    """Executes the creation of all tables registered in the Base metadata registry.
    
    Uses ``checkfirst=True`` so that tables and indexes that already exist in the
    database file are silently skipped rather than raising ``OperationalError``.
    """
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Executed create_all for database schemas.")
    except Exception as e:
        logger.error(f"Failed to execute create_all_tables: {e}")
        raise



def drop_all_tables() -> None:
    """Drops all tables registered in the Base metadata registry.

    Warning: This wipes the entire local database. Use for testing/resetting only.
    """
    with _db_lock:
        logger.warning("Dropping all database tables.")
        try:
            Base.metadata.drop_all(bind=engine)
            logger.info("All tables dropped successfully.")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise


def dispose_engine() -> None:
    """Disposes all active connections in the connection pool gracefully."""
    logger.info("Disposing database engine connection pools.")
    try:
        engine.dispose()
    except Exception as e:
        logger.error(f"Failed to dispose engine connection pools: {e}")


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of database operations.

    Ensures that errors automatically trigger rollbacks, and that resources
    are cleaned up and closed under all circumstances.

    Yields:
        Session: A database session instance.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        logger.error(f"Database error in transaction scope, triggering rollback: {e}")
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a thread-safe database session.

    Cleans up and closes the connection when the request lifecyle ends.

    Yields:
        Session: Active database session.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database dependency session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def health_check() -> bool:
    """Verifies connection health to the local SQLite database.

    Executes a basic 'SELECT 1' test query.

    Returns:
        bool: True if connection is healthy, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
