"""add categories to scenarios

Revision ID: 3c1f0b7a9d24
Revises: 8ff9b412282f
Create Date: 2026-11-16 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3c1f0b7a9d24'
down_revision: Union[str, Sequence[str], None] = '8ff9b412282f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('scenarios', schema=None) as batch_op:
        batch_op.add_column(sa.Column('categories', sa.String(length=300), server_default='', nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('scenarios', schema=None) as batch_op:
        batch_op.drop_column('categories')
