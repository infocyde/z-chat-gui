"""add lorebooks

Revision ID: c4a8e1f2d907
Revises: b7e2d4c81f30
Create Date: 2026-11-18 10:00:00.000000

Written by hand: Alembic can't autogenerate the FTS5 table or its triggers (and env.py tells
autogenerate to ignore the lore_fts* tables, so it won't try to drop them either).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4a8e1f2d907'
down_revision: Union[str, Sequence[str], None] = 'b7e2d4c81f30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'scenario_lore',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('scenario_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('keys', sa.String(length=500), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['scenario_id'], ['scenarios.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_scenario_lore_scenario_id', 'scenario_lore', ['scenario_id'])

    op.create_table(
        'lore_entries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_lore_entries_conversation_id', 'lore_entries', ['conversation_id'])

    op.create_table(
        'lore_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entry_id', sa.Integer(), nullable=False),
        sa.Column('message_id', sa.Integer(), nullable=True),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('keys', sa.String(length=500), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['entry_id'], ['lore_entries.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_lore_versions_entry_id', 'lore_versions', ['entry_id'])
    op.create_index('ix_lore_versions_message_id', 'lore_versions', ['message_id'])

    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.add_column(sa.Column('lore_used', sa.Text(), server_default='', nullable=False))
        batch_op.add_column(sa.Column('lore_note', sa.Text(), server_default='', nullable=False))

    # Full-text index of every lore version (rowid = lore_versions.id), kept in sync by triggers.
    # Triggers also fire on cascade deletes, so deleting a chat cleans the index too.
    op.execute("CREATE VIRTUAL TABLE lore_fts USING fts5("
               "name, keys, text, tokenize='porter unicode61 remove_diacritics 2')")
    op.execute("""
        CREATE TRIGGER lore_versions_ai AFTER INSERT ON lore_versions BEGIN
            INSERT INTO lore_fts(rowid, name, keys, text) VALUES (new.id, new.name, new.keys, new.text);
        END""")
    op.execute("""
        CREATE TRIGGER lore_versions_ad AFTER DELETE ON lore_versions BEGIN
            DELETE FROM lore_fts WHERE rowid = old.id;
        END""")
    op.execute("""
        CREATE TRIGGER lore_versions_au AFTER UPDATE OF name, keys, text ON lore_versions BEGIN
            DELETE FROM lore_fts WHERE rowid = old.id;
            INSERT INTO lore_fts(rowid, name, keys, text) VALUES (new.id, new.name, new.keys, new.text);
        END""")


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP TRIGGER IF EXISTS lore_versions_au")
    op.execute("DROP TRIGGER IF EXISTS lore_versions_ad")
    op.execute("DROP TRIGGER IF EXISTS lore_versions_ai")
    op.execute("DROP TABLE IF EXISTS lore_fts")
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_column('lore_note')
        batch_op.drop_column('lore_used')
    op.drop_index('ix_lore_versions_message_id', table_name='lore_versions')
    op.drop_index('ix_lore_versions_entry_id', table_name='lore_versions')
    op.drop_table('lore_versions')
    op.drop_index('ix_lore_entries_conversation_id', table_name='lore_entries')
    op.drop_table('lore_entries')
    op.drop_index('ix_scenario_lore_scenario_id', table_name='scenario_lore')
    op.drop_table('scenario_lore')
