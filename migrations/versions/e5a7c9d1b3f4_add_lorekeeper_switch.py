"""add the Lorekeeper switch (conversations.lore_on)

Revision ID: e5a7c9d1b3f4
Revises: d91b3c5e7a12
Create Date: 2026-11-18

"""
from typing import Sequence, Union

from alembic import op

revision: str = 'e5a7c9d1b3f4'
down_revision: Union[str, Sequence[str], None] = 'd91b3c5e7a12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Plain ALTER TABLE (not batch mode), like the embeddings migration.
    op.execute("ALTER TABLE conversations ADD COLUMN lore_on BOOLEAN NOT NULL DEFAULT 1")
    # Existing plain chats (no scenario, nothing in their lorebook) start with it off, like new ones.
    op.execute("UPDATE conversations SET lore_on = 0 WHERE scenario_id IS NULL "
               "AND id NOT IN (SELECT conversation_id FROM lore_entries)")


def downgrade() -> None:
    op.execute("ALTER TABLE conversations DROP COLUMN lore_on")
