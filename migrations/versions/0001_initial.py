"""Create ticket and prediction ledger."""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None


def upgrade():
    op.create_table("tickets",
                    sa.Column("id", sa.String(36), primary_key=True),
                    sa.Column("context", sa.JSON(), nullable=False))
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticket_id", sa.String(36), sa.ForeignKey("tickets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.String(36), nullable=False),
        sa.Column("message", sa.JSON(), nullable=False),
        sa.Column("risk", sa.Float(), nullable=False),
        sa.Column("level", sa.String(12), nullable=False),
        sa.Column("alert", sa.Boolean(), nullable=False),
        sa.Column("version", sa.String(32), nullable=False),
        sa.Column("decision", sa.JSON(), nullable=False),
        sa.UniqueConstraint("ticket_id", "position"),
        sa.UniqueConstraint("ticket_id", "message_id"),
        sa.CheckConstraint("position > 0"),
        sa.CheckConstraint("risk >= 0 AND risk <= 1"),
    )
    op.create_index("ix_predictions_alert", "predictions", ["alert"])


def downgrade():
    op.drop_table("predictions")
    op.drop_table("tickets")
