"""add lore embeddings

Revision ID: d91b3c5e7a12
Revises: c4a8e1f2d907
Create Date: 2026-11-18 11:00:00.000000

Plain ALTER TABLE on purpose: Alembic's batch mode may rebuild the table, and rebuilding
lore_versions would drop the triggers that keep the lore_fts index in sync.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd91b3c5e7a12'
down_revision: Union[str, Sequence[str], None] = 'c4a8e1f2d907'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TABLE lore_versions ADD COLUMN embedding BLOB")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("ALTER TABLE lore_versions DROP COLUMN embedding")
