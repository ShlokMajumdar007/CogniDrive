from datetime import datetime, timezone
import uuid as uuid_pkg
from typing import Optional

from sqlalchemy import Boolean, DateTime, String, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, registry


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy database models.

    Uses SQLAlchemy 2.0 Declarative style and standard type mappings.
    """

    registry = registry()


class TimestampMixin:
    """SQLAlchemy mixin class to add created_at and updated_at timestamps.

    Timestamps are stored in UTC format.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default=text("CURRENT_TIMESTAMP"),
        doc="UTC Timestamp showing when the record was created",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=text("CURRENT_TIMESTAMP"),
        doc="UTC Timestamp showing when the record was last updated",
    )


class UUIDMixin:
    """SQLAlchemy mixin class to add a unique UUID4 column."""

    uuid: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
        default=lambda: str(uuid_pkg.uuid4()),
        doc="Standard UUID4 string for secure external resource identification",
    )


class SoftDeleteMixin:
    """SQLAlchemy mixin class to support soft-deletion of records.

    Allows records to be flagged as deleted rather than permanently purged from disk.
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("0"),
        doc="Boolean flag representing whether this record was soft-deleted",
    )

    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
        doc="UTC Timestamp showing when the record was soft-deleted",
    )

    def soft_delete(self) -> None:
        """Flags the record as deleted and sets the deletion timestamp to current UTC time."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
