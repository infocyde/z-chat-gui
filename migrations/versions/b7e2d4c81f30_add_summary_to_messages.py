"""add summary to messages

Revision ID: b7e2d4c81f30
Revises: 3c1f0b7a9d24
Create Date: 2026-11-16 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e2d4c81f30'
down_revision: Union[str, Sequence[str], None] = '3c1f0b7a9d24'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.add_column(sa.Column('summary', sa.Text(), server_default='', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('messages', schema=None) as batch_op:
        batch_op.drop_column('summary')
