import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
import pytest

from escalation.api import create_app
from escalation.policy import Policy


def migrate(url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")


@pytest.fixture(params=["sqlite", "postgres"])
def service(request, tmp_path, monkeypatch):
    url = ("sqlite:///" + str(tmp_path / "tickets.db") if request.param == "sqlite"
           else os.getenv("TEST_POSTGRES_URL"))
    if not url:
        pytest.skip("PostgreSQL integration requires TEST_POSTGRES_URL")
    migrate(url, monkeypatch)
    app = create_app(url, Path("artifacts"))
    with TestClient(app) as client:
        yield client, url
    app.state.engine.dispose()


def payload(at, text="Where is my refund?", role="customer"):
    return {"message_id": str(uuid4()), "message": {"at": at.isoformat(), "text": text, "role": role}}


def test_replay_persistence_and_idempotency(service):
    client, url = service
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/openapi.json").status_code == 200
    sample = client.get("/api/demo").json()[0]
    identifier = client.post("/api/tickets", json=sample["context"]).json()["id"]
    endpoint = f"/api/tickets/{identifier}"
    scores = []
    for pos, message in enumerate(sample["messages"], 1):
        data = {"message_id": str(uuid4()), "message": message}
        response = client.post(endpoint + "/messages", json=data)
        assert response.status_code == 200, response.text
        prediction = response.json()
        assert prediction["position"] == pos
        assert 0 <= prediction["risk"] <= 1
        assert client.post(endpoint + "/messages", json=data).json() == prediction
        scores.append(prediction)
    assert len({r["risk"] for r in scores}) > 1
    assert any(r["alert"] for r in scores)
    assert client.get(endpoint + "/recommendation").json() == scores[-1]
    assert client.get("/api/model/info").json()["version"] == scores[-1]["version"]
    assert client.get("/api/metrics").json()["holdout"]["baseline"]["tickets"] == 240
    assert client.get("/api/alerts").json()
    with TestClient(create_app(url)) as restarted:
        assert restarted.get(endpoint + "/risk-history").json() == scores


def test_bad_requests_leave_history_unchanged(service):
    client, _ = service
    now = datetime.now(timezone.utc)
    identifier = client.post("/api/tickets", json={"category":"refund"}).json()["id"]
    path = f"/api/tickets/{identifier}"
    data = payload(now)
    assert client.post(path + "/messages", json=data).status_code == 200
    assert client.post(path + "/messages", json=payload(now)).status_code == 409
    data["message"]["text"] = "Changed payload"
    assert client.post(path + "/messages", json=data).status_code == 409
    assert client.post(path + "/messages", json=payload(now, "")).status_code == 422
    assert client.get(f"/api/tickets/{uuid4()}").status_code == 404
    assert len(client.get(path + "/risk-history").json()) == 1


def test_postgres_duplicate_concurrent_submissions(service):
    client, url = service
    if not url.startswith("postgres"):
        pytest.skip("PostgreSQL row locking")
    identifier = client.post("/api/tickets", json={"category":"billing"}).json()["id"]
    endpoint = f"/api/tickets/{identifier}"
    data = payload(datetime.now(timezone.utc))
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: client.post(endpoint + "/messages", json=data), range(4)))
    assert [r.status_code for r in responses] == [200] * 4
    assert all(r.json() == responses[0].json() for r in responses)
    assert len(client.get(endpoint + "/risk-history").json()) == 1


def test_policy_cooldown_and_severity():
    policy = Policy(0.5, cooldown_minutes=45)
    now = datetime.now(timezone.utc)
    assert not policy.decide(0.49, now, None, None)["alert"]
    assert policy.decide(0.5, now, None, None)["alert"]
    recent = ("high", now)
    assert not policy.decide(0.95, now + timedelta(minutes=44), {"risk": 0.5}, recent)["alert"]
    critical = policy.decide(0.95, now + timedelta(minutes=45), {"risk": 0.5}, recent)
    assert critical["alert"] and critical["rapid_increase"]
    assert not policy.decide(0.60, now + timedelta(minutes=60), None, recent)["alert"]
