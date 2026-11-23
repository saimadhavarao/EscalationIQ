from sqlalchemy import JSON, Boolean, CheckConstraint, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TicketRow(Base):
    __tablename__ = "tickets"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    context: Mapped[dict] = mapped_column(JSON)


class PredictionRow(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("ticket_id", "position"),
        UniqueConstraint("ticket_id", "message_id"),
        CheckConstraint("position > 0"),
        CheckConstraint("risk >= 0 AND risk <= 1"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticket_id: Mapped[str] = mapped_column(ForeignKey("tickets.id", ondelete="CASCADE"))
    position: Mapped[int]
    message_id: Mapped[str] = mapped_column(String(36))
    message: Mapped[dict] = mapped_column(JSON)
    risk: Mapped[float] = mapped_column(Float)
    level: Mapped[str] = mapped_column(String(12))
    alert: Mapped[bool] = mapped_column(Boolean, index=True)
    version: Mapped[str] = mapped_column(String(32))
    decision: Mapped[dict] = mapped_column(JSON)
