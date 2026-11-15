"""Postgres cache of decomposer plans, keyed by the rewritten query."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from lightrag_wrapper.config import LightRAGConfig
from lightrag_wrapper.log import get_logger
from lightrag_wrapper.models import QueryPlanNode

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy base for wrapper-owned tables."""


class DecomposerPlanRow(Base):
    """One stored plan. The primary key is the rewritten query."""

    __tablename__ = "decomposer_plan_cache"

    rewritten_query: Mapped[str] = mapped_column(Text, primary_key=True)
    plan: Mapped[list] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class PlanCache:
    """Fetch and store decomposer plans in Postgres."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)

    async def get(self, rewritten_query: str) -> list[QueryPlanNode] | None:
        """Return the stored plan for this rewritten query, or None on a miss."""
        key = rewritten_query.strip()
        if not key:
            return None
        async with self._sessions() as session:
            row = await session.get(DecomposerPlanRow, key)
        if row is None:
            return None
        try:
            return [QueryPlanNode.model_validate(item) for item in row.plan]
        except ValueError:
            logger.warning("Ignoring an unreadable cached decomposer plan", query=key[:120])
            return None

    async def put(self, rewritten_query: str, nodes: list[QueryPlanNode]) -> None:
        """Store a plan under the rewritten query. A duplicate key leaves the first row."""
        key = rewritten_query.strip()
        if not key or not nodes:
            return
        payload = [node.model_dump(by_alias=True) for node in nodes]
        async with self._sessions() as session:
            session.add(DecomposerPlanRow(rewritten_query=key, plan=payload))
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()

    async def aclose(self) -> None:
        await self._engine.dispose()


def _database_url(config: LightRAGConfig) -> URL:
    return URL.create(
        "postgresql+asyncpg",
        username=config.postgres_user,
        password=config.postgres_password or "",
        host=config.postgres_host,
        port=config.postgres_port,
        database=config.postgres_database,
    )


async def open_plan_cache(config: LightRAGConfig) -> PlanCache:
    """Connect and create the plan table when it is not already there."""
    engine = create_async_engine(_database_url(config), pool_pre_ping=True)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    except Exception:
        await engine.dispose()
        raise
    logger.info("Decomposer plan cache is ready")
    return PlanCache(engine)
