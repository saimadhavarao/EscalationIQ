import json
import logging
import os
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from .data import Context, Message
from .db import PredictionRow, TicketRow
from .model import Predictor
from .policy import Policy

log = logging.getLogger("uvicorn.error")


class Incoming(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: UUID
    message: Message


class Prediction(BaseModel):
    ticket_id: str
    position: int
    message: Message
    risk: float
    level: str
    alert: bool
    version: str
    action: str
    delta: float
    rapid_increase: bool
    alert_reason: str
    signals: list[str]
    contributions: list[dict[str, str | float]]
    latency_ms: float


def render(row: PredictionRow):
    return {"ticket_id": row.ticket_id, "position": row.position, "message": row.message,
            "risk": row.risk, "level": row.level, "alert": row.alert, "version": row.version,
            **row.decision}


def history(db, ticket_id):
    return list(db.scalars(select(PredictionRow).where(PredictionRow.ticket_id == ticket_id)
                           .order_by(PredictionRow.position)))


def require_ticket(db, ticket_id, lock=False):
    ticket = db.get(TicketRow, str(ticket_id), with_for_update=lock)
    if ticket is None:
        raise HTTPException(404, "Ticket does not exist")
    return ticket


def create_app(database_url: str | None = None, artifact_dir: Path | None = None):
    artifact_dir = artifact_dir or Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    engine = create_engine(database_url or os.getenv("DATABASE_URL", "sqlite:///escalation.db"),
                           pool_pre_ping=True)
    predictor = Predictor(artifact_dir / "model.joblib")
    policy = Policy(float(os.getenv("HIGH_THRESHOLD", str(predictor.info["threshold"]))),
                    int(os.getenv("ALERT_COOLDOWN_MINUTES", "45")))
    if not 0 < policy.high <= 1 or policy.cooldown_minutes < 0:
        raise ValueError("Invalid alert policy configuration")
    app = FastAPI(title="EscalationIQ", version="1.0")
    app.state.engine = engine

    @app.get("/health")
    def health():
        with Session(engine) as db:
            db.execute(text("SELECT 1"))
            db.execute(select(TicketRow.id).limit(1))
        return {"status": "ok", "model_version": predictor.info["version"]}

    @app.post("/api/tickets", status_code=201)
    def create_ticket(context: Context):
        identifier = str(uuid4())
        with Session(engine) as db, db.begin():
            db.add(TicketRow(id=identifier, context=context.model_dump()))
        return {"id": identifier, "context": context}

    @app.post("/api/tickets/{ticket_id}/messages", response_model=Prediction)
    def add_message(ticket_id: UUID, incoming: Incoming):
        with Session(engine) as db, db.begin():
            # Serializes message insertion and policy decisions for a PostgreSQL ticket.
            ticket = require_ticket(db, ticket_id, lock=True)
            previous_rows = history(db, ticket.id)
            payload = incoming.message.model_dump(mode="json")
            for existing in previous_rows:
                if existing.message_id == str(incoming.message_id):
                    if existing.message != payload:
                        raise HTTPException(409, "Message ID already used with different content")
                    return render(existing)
            if len(previous_rows) >= 250:
                raise HTTPException(409, "Ticket has reached 250 messages")
            previous = previous_rows[-1] if previous_rows else None
            messages = [Message.model_validate(r.message) for r in previous_rows]
            if messages and incoming.message.at <= messages[-1].at:
                raise HTTPException(409, "Message timestamp must follow the previous message")
            score = predictor.score([*messages, incoming.message], Context.model_validate(ticket.context))
            last_alert = next((r for r in reversed(previous_rows) if r.alert), None)
            decision = policy.decide(score["risk"], incoming.message.at,
                                     {"risk": previous.risk} if previous else None,
                                     (last_alert.level, datetime.fromisoformat(last_alert.message["at"]))
                                     if last_alert else None)
            record = PredictionRow(ticket_id=ticket.id, position=len(messages) + 1,
                                   message_id=str(incoming.message_id), message=payload,
                                   version=predictor.info["version"], risk=score.pop("risk"),
                                   level=decision.pop("level"), alert=decision.pop("alert"),
                                   decision={**score, **decision})
            db.add(record)
            log.info(json.dumps({"event": "risk_scored", "ticket_id": ticket.id,
                                 "position": record.position, "level": record.level,
                                 "alert": record.alert, "model_version": record.version}))
            return render(record)

    @app.get("/api/tickets")
    def list_tickets(limit: int = 100, offset: int = 0):
        if not 1 <= limit <= 200 or offset < 0:
            raise HTTPException(422, "Invalid pagination")
        with Session(engine) as db:
            tickets = list(db.scalars(select(TicketRow).order_by(TicketRow.id).offset(offset).limit(limit)))
            result = []
            for ticket in tickets:
                records = history(db, ticket.id)
                result.append({"id": ticket.id, "context": ticket.context, "message_count": len(records),
                               "latest": render(records[-1]) if records else None,
                               "alerted": any(r.alert for r in records)})
        return result

    @app.get("/api/tickets/{ticket_id}")
    def ticket_detail(ticket_id: UUID):
        with Session(engine) as db:
            ticket = require_ticket(db, ticket_id)
            return {"id": ticket.id, "context": ticket.context,
                    "history": [render(r) for r in history(db, ticket.id)]}

    @app.get("/api/tickets/{ticket_id}/risk-history", response_model=list[Prediction])
    def risk_history(ticket_id: UUID):
        return ticket_detail(ticket_id)["history"]

    @app.get("/api/tickets/{ticket_id}/recommendation", response_model=Prediction | None)
    def recommendation(ticket_id: UUID):
        records = risk_history(ticket_id)
        return records[-1] if records else None

    @app.get("/api/alerts", response_model=list[Prediction])
    def alerts():
        with Session(engine) as db:
            return [render(r) for r in db.scalars(select(PredictionRow)
                    .where(PredictionRow.alert.is_(True)).order_by(PredictionRow.id.desc()).limit(200))]

    @app.get("/api/model/info")
    def info():
        return {**predictor.info, "policy": {"high": policy.high, "critical": policy.critical,
                                            "cooldown_minutes": policy.cooldown_minutes}}

    @app.get("/api/metrics")
    def metrics():
        return json.loads((artifact_dir / "evaluation-results.json").read_text(encoding="utf-8"))

    @app.get("/api/demo")
    def demo():
        return json.loads((artifact_dir / "demo.json").read_text(encoding="utf-8"))

    return app
