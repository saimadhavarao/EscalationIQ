from datetime import timedelta

import numpy as np
import pytest
from pydantic import ValidationError

from escalation.data import Context, Message, Ticket, audit_splits, generate
from escalation.evaluate import operating_threshold, summarize
from escalation.features import structured
from escalation.model import Predictor
from escalation.train import prefixes, run


@pytest.fixture(scope="module")
def splits():
    return generate()


def test_chronological_disjoint_families_and_entities(splits):
    audit_splits(splits)
    train, validation, holdout = splits.values()
    assert train[-1].messages[-1].at < validation[0].messages[0].at
    assert validation[-1].messages[-1].at < holdout[0].messages[0].at
    texts = [{m.text for t in group for m in t.messages} for group in splits.values()]
    assert all(not (left & right) for i, left in enumerate(texts) for right in texts[i + 1:])
    with pytest.raises(ValueError, match="Duplicate"):
        audit_splits({"train": train, "test": train[:1]})


def test_dataset_validation(splits):
    ticket = splits["train"][0].model_dump()
    with pytest.raises(ValidationError, match="strictly increase"):
        Ticket.model_validate({**ticket, "messages": list(reversed(ticket["messages"]))})
    with pytest.raises(ValidationError, match="Event"):
        Ticket.model_validate({**ticket, "event": 1})
    with pytest.raises(ValidationError):
        Message(role="agent", text="  ", at=ticket["messages"][0]["at"])


def test_future_content_and_labels_cannot_change_prefix_score(splits):
    from pathlib import Path

    predictor = Predictor(Path("artifacts/model.joblib"))
    ticket = next(t for t in splits["train"] if t.event is not None)
    initial = ticket.messages[:2]
    edited = Ticket.model_validate({
        **ticket.model_dump(),
        "messages": [*initial, Message(role="agent", text="Supervisor took over",
                                       at=initial[-1].at + timedelta(minutes=90))],
        "event": None,
    })
    assert len(prefixes(ticket)) == ticket.event - 1
    a = predictor.score(prefixes(ticket)[1], ticket.context)
    b = predictor.score(prefixes(edited)[1], edited.context)
    assert a["risk"] == b["risk"]
    assert a["contributions"] == b["contributions"]
    assert "event" not in structured(initial, ticket.context)


def test_behavior_and_missing_history(splits):
    at = splits["train"][0].messages[0].at
    messages = [
        Message(role="customer", text="Where is my refund?", at=at),
        Message(role="agent", text="Try reopening the page.", at=at + timedelta(minutes=12)),
        Message(role="customer", text="Where is my refund? Still waiting.", at=at + timedelta(minutes=14)),
    ]
    values = structured(messages, Context(category="refund", prior_escalations=0))
    assert values["average_reply_minutes"] == 12
    assert values["repeat_after_step"] == 1
    assert values["open_questions"] == 1
    assert values["tone_trend"] > 0
    assert values["prior_tickets_missing"] == 1
    assert values["prior_escalations_missing"] == 0


def test_evaluation_counts_strictly_before_event(splits):
    positive = next(t for t in splits["test"] if t.event is not None)
    negative = next(t for t in splits["test"] if t.event is None)
    early = np.zeros(positive.event - 1)
    early[-1] = 0.8
    false = np.full(len(negative.messages), 0.7)
    metrics = summarize([positive, negative], [early, false], 0.75)
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]
    assert metrics["detected_at_least"] == {"1": 1, "2": 0, "3": 0}
    assert metrics["lead_minutes"][0] > 0
    assert operating_threshold([positive, negative], [early, false]) > 0.8


def test_training_and_reload(tmp_path):
    run(tmp_path, 42, structured_only=True, counts=(100, 70, 70))
    model = Predictor(tmp_path / "model.joblib")
    ticket = generate(42, (2, 2, 2))["train"][0]
    assert 0 < model.score(ticket.messages[:1], ticket.context)["risk"] < 1
    if model.model.kind == "baseline":
        train_text = {m.text for t in generate(42, (100, 70, 70))["train"] for m in t.messages}
        assert not any("advise" in line for line in train_text)
        assert "advise" not in model.model.vectorizer.vocabulary_
