"""Prefix-only inputs shared by training and inference."""
import re
from functools import lru_cache

import numpy as np

from .data import Context, Message, TOPICS

EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
NEGATIVE = {"still", "frustrating", "upsetting", "nothing", "waiting", "broken", "locked"}
POSITIVE = {"thanks", "helps", "okay", "understand"}
HISTORY = ["prior_tickets", "prior_escalations", "tenure_days"]
KEYS = [
    "message_count", "customer_agent_ratio", "open_questions", "repeat_requests",
    "average_reply_minutes", "duration_minutes", "suggested_steps", "repeat_after_step",
    "since_agent_minutes", "tone_trend", "latest_tone",
] + [f"{key}{suffix}" for key in HISTORY for suffix in ("", "_missing")] + [
    f"category_{category}" for category in TOPICS
]


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def structured(messages: list[Message], context: Context) -> dict[str, float]:
    customers, delays, tones = [], [], []
    last_customer = last_agent = None
    questions = repeats = steps = repeat_after_step = 0
    step_pending = False
    for message in messages:
        tokens = words(message.text)
        if message.role == "customer":
            repeated = bool(customers and any(
                len(tokens & earlier) / max(1, len(tokens | earlier)) >= 0.55 for earlier in customers[-4:]
            ))
            repeats += repeated
            repeat_after_step += repeated and step_pending
            questions += message.text.count("?")
            customers.append(tokens)
            tones.append(len(tokens & NEGATIVE) - len(tokens & POSITIVE))
            if last_customer is None:
                last_customer = message.at
            step_pending = False
        else:
            if last_customer is not None:
                delays.append((message.at - last_customer).total_seconds() / 60)
            last_customer = None
            last_agent = message.at
            questions = 0
            step_pending = bool(tokens & {"try", "refresh", "restart", "retry"})
            steps += step_pending
    now = messages[-1].at
    values = [
        len(messages), len(customers) / max(1, len(messages) - len(customers)), questions, repeats,
        float(np.mean(delays)) if delays else 0, (now - messages[0].at).total_seconds() / 60,
        steps, repeat_after_step, (now - (last_agent or messages[0].at)).total_seconds() / 60,
        float(np.mean(tones[-2:]) - tones[0]) if tones else 0, tones[-1] if tones else 0,
    ]
    for key in HISTORY:
        value = getattr(context, key)
        values.extend((value or 0, value is None))
    values.extend(context.category == category for category in TOPICS)
    return dict(zip(KEYS, map(float, values), strict=True))


@lru_cache(maxsize=1)
def encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING, revision=REVISION, device="cpu")


@lru_cache(maxsize=8192)
def embedding(text: str):
    return encoder().encode(text, normalize_embeddings=True)


def warm_embeddings(texts: set[str]) -> None:
    # A training run uses a small, explicit vocabulary of complete utterances.
    ordered = sorted(texts)
    _training_vectors.update(zip(ordered, encoder().encode(ordered, normalize_embeddings=True), strict=True))


_training_vectors = {}


def semantic(messages: list[Message]) -> np.ndarray:
    vectors = np.stack([_training_vectors[m.text] if m.text in _training_vectors else embedding(m.text)
                        for m in messages])
    customer = [v for m, v in zip(messages, vectors) if m.role == "customer"]
    neighbors = np.sum(vectors[1:] * vectors[:-1], axis=1)
    mismatch = [1 - float(neighbors[i]) for i in range(len(neighbors))
                if messages[i].role == "customer" and messages[i + 1].role == "agent"]
    return np.r_[vectors.mean(axis=0), vectors[-1],
                 float(neighbors.mean()) if len(neighbors) else 0,
                 float(customer[-1] @ customer[-2]) if len(customer) > 1 else 0,
                 float(np.mean(mismatch)) if mismatch else 0]


def signals(values: dict[str, float]) -> list[str]:
    evidence = []
    if values["repeat_requests"]:
        evidence.append(f'{int(values["repeat_requests"])} customer messages resembled earlier requests')
    if values["tone_trend"] > 0:
        evidence.append("Recent customer messages used more negative language than the opening")
    if values["repeat_after_step"]:
        evidence.append(f'{int(values["repeat_after_step"])} repeated requests followed a suggested step')
    if values["average_reply_minutes"]:
        evidence.append(f'Mean observed agent reply time was {values["average_reply_minutes"]:.1f} minutes')
    return evidence
