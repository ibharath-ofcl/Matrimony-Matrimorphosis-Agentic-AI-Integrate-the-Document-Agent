"""create users, tokens, threads tables

Revision ID: 001_create_users
Revises: None
"""
from alembic import op
import sqlalchemy as sa

revision = "001_create_users"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True, nullable=False),
        sa.Column("username", sa.String(), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("created", sa.Float(), nullable=False),
    )
    op.create_table(
        "tokens",
        sa.Column("token", sa.String(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("created", sa.Float(), nullable=False),
    )
    op.create_table(
        "threads",
        sa.Column("thread_id", sa.String(), primary_key=True, nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("threads")
    op.drop_table("tokens")
    op.drop_table("users")

