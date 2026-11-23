"""Deterministic simulation. Event timing is separate from message content."""
from datetime import datetime, timedelta, timezone
from random import Random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, AwareDatetime, model_validator

TOPICS = {
    "billing": "duplicate invoice", "refund": "missing refund", "technical": "sync error",
    "cancellation": "subscription cancellation", "recurring": "recurring outage",
    "defect": "broken replacement", "delivery": "undelivered parcel",
    "access": "locked account", "routine": "receipt request",
}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    role: Literal["customer", "agent"]
    text: str = Field(min_length=1, max_length=4000)
    at: AwareDatetime


class Context(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: str = Field(min_length=1, max_length=40)
    prior_tickets: int | None = Field(None, ge=0, le=10000)
    prior_escalations: int | None = Field(None, ge=0, le=10000)
    tenure_days: int | None = Field(None, ge=0, le=50000)


class Ticket(BaseModel):
    id: str
    customer: str
    cohort: str
    context: Context
    messages: list[Message] = Field(min_length=2)
    event: int | None

    @model_validator(mode="after")
    def validate_timeline(self):
        if self.event is not None and not 2 <= self.event <= len(self.messages):
            raise ValueError("Event must follow at least one observable message")
        if any(a.at >= b.at for a, b in zip(self.messages, self.messages[1:])):
            raise ValueError("Message times must strictly increase")
        return self


# Each cohort has a separate phrasing family. These choices are never features.
COPY = {
    "train": (
        ["Can you help me with {issue}?", "I'm contacting you about {issue}."],
        ["I still have the {issue}.", "This is frustrating. The {issue} is still happening."],
        ["I can check {issue} for you.", "Try refreshing and let me know if that works."],
        ["Thanks, that helps.", "I understand. What happens next?"],
    ),
    "validation": (
        ["Could someone look into {issue}?", "I need some help with {issue}."],
        ["This is getting frustrating. Nothing changed.", "Still waiting on {issue}. Any progress?"],
        ["I'll verify the details of {issue}.", "Try signing back in before the next check."],
        ["Okay, I can do that.", "Thanks for following up."],
    ),
    "test": (
        ["I'm following up regarding {issue}.", "Please advise on {issue}."],
        ["We are back where we started with {issue}.", "This is upsetting. I keep running into {issue}."],
        ["Let me confirm what happened with {issue}.", "Try opening a fresh session and check again."],
        ["That makes sense.", "I'll keep an eye on it. Thanks."],
    ),
}


def generate(seed: int = 42, counts: tuple[int, int, int] = (480, 180, 240)) -> dict[str, list[Ticket]]:
    rng = Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    splits = {}
    serial = 0
    seen = set()
    for cohort, count in zip(COPY, counts, strict=True):
        splits[cohort] = []
        opening, repeat, reply, neutral = COPY[cohort]
        while len(splits[cohort]) < count:
            category, issue = rng.choice(list(TOPICS.items()))
            friction = rng.betavariate(2, 3)
            urgency = rng.random()
            prior = rng.randrange(7)
            unresolved = 0
            messages = []
            at = start + timedelta(days=serial)
            event = None
            for index in range(rng.randint(5, 13)):
                customer = index == 0 or rng.random() < (0.23 if messages[-1].role == "customer" else 0.82)
                blocked = rng.random() < min(0.93, 0.15 + 0.66 * friction + 0.025 * index)
                if customer and index > 0:
                    unresolved = max(0, unresolved + (1 if blocked else -1))
                lines = opening if index == 0 else (repeat if blocked else neutral) if customer else reply
                elapsed = rng.randint(1, 7) + (round(19 * friction) if not customer else 0)
                at += timedelta(minutes=elapsed)
                messages.append(Message(role="customer" if customer else "agent",
                                        text=rng.choice(lines).format(issue=issue), at=at))
                hazard = 0.004 + 0.13 * friction ** 2 + 0.035 * urgency + 0.017 * unresolved + 0.003 * prior
                if category == "routine":
                    hazard *= 0.4
                if index >= 3 and rng.random() < hazard:
                    event = index + 1
                    break
            signature = tuple((m.role, m.text.casefold()) for m in messages)
            if signature in seen:
                continue
            seen.add(signature)
            context = Context(
                category=category,
                prior_tickets=prior if rng.random() > 0.22 else None,
                prior_escalations=rng.randint(0, min(2, prior)) if rng.random() > 0.3 else None,
                tenure_days=rng.randint(5, 1200) if rng.random() > 0.3 else None,
            )
            splits[cohort].append(Ticket(id=f"T{serial:05}", customer=f"C{serial:05}", cohort=cohort,
                                         context=context, messages=messages, event=event))
            serial += 1
    audit_splits(splits)
    return splits


def audit_splits(splits: dict[str, list[Ticket]]) -> None:
    ids, customers, transcripts = set(), set(), set()
    for tickets in splits.values():
        for ticket in tickets:
            signature = tuple((m.role, m.text.casefold()) for m in ticket.messages)
            if ticket.id in ids or ticket.customer in customers or signature in transcripts:
                raise ValueError("Duplicate ticket, customer, or transcript")
            ids.add(ticket.id)
            customers.add(ticket.customer)
            transcripts.add(signature)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Export simulated tickets with explicit outcome fields")
    parser.add_argument("--output", type=Path, default=Path("generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--counts", type=int, nargs=3, default=(480, 180, 240))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for cohort, tickets in generate(args.seed, tuple(args.counts)).items():
        with (args.output / f"{cohort}.jsonl").open("w", encoding="utf-8") as stream:
            for ticket in tickets:
                stream.write(ticket.model_dump_json() + "\n")
