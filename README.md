# EscalationIQ

EscalationIQ scores support conversations after each incoming message. It gives operators a ticket queue, a history of risk and alert decisions, and recommendations for human specialist review. The replay uses the same API as any other ticket submission.

```mermaid
flowchart LR
  A["Incoming message"] --> B["Ordered conversation prefix"]
  B --> C["Calibrated classifier"]
  C --> D["Advisory alert policy"]
  D --> E["Risk ledger and dashboard"]
```

FastAPI and SQLAlchemy serve the model. PostgreSQL persists messages and decisions with Alembic migrations. React, TypeScript and Recharts power the dashboard. Python, scikit-learn, joblib and Sentence Transformers run the experiment.

## Windows quick start

Extract the ZIP and open the inner `EscalationIQ` folder in VS Code. With Docker Desktop running, use PowerShell.

```powershell
docker compose up --build -d --wait
Invoke-RestMethod http://localhost:8000/health
Start-Process http://localhost:5173
```

Open **Live replay**, select the first scenario and step through its messages. Or prepopulate the queue with `docker compose exec api python -m escalation.seed`. The model is bundled. No API key or database installation is needed.

## Dataset

The generator creates 900 fictional conversations across nine support categories. It samples escalation timing from a latent hazard driven by friction, urgency and unresolved issues. The model receives prefix messages and creation-time context. Ticket IDs, cohort IDs, latent variables and outcomes stay outside the feature input. Different chronological cohorts use distinct wording. [Data provenance and leakage audit](docs/EVALUATION.md)

Export the generated dataset to JSONL with `python -m escalation.data --seed 42`. The six replay examples in `artifacts/demo.json` were generated separately and curated for the demo.

## Executed experiment

| Candidate | Validation F1 | Holdout precision | Holdout recall | Holdout F1 | Holdout PR-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF logistic regression (deployed) | 0.408 | 40.0% | 15.6% | 0.224 | 0.371 |
| Structured gradient boosting | 0.314 | 50.0% | 20.8% | 0.294 | 0.423 |
| MiniLM hybrid logistic regression | 0.255 | 37.5% | 19.5% | 0.256 | 0.358 |

The deployed model produced 18 false alerts across 240 holdout tickets and detected 12 of 77 escalations ahead of the event. The median lead was four messages and 29 minutes among those 12 detections. Its validation F1 was highest under the preset alert budget. The structured model performed better on holdout. All metrics describe the synthetic benchmark.

## API example

```powershell
$base = 'http://localhost:8000'
$ticket = Invoke-RestMethod -Method Post -Uri "$base/api/tickets" -ContentType 'application/json' -Body '{"category":"refund"}'
$body = @{
  message_id = [guid]::NewGuid().ToString()
  message = @{ role = 'customer'; text = 'Where is my refund?'; at = [DateTimeOffset]::UtcNow.ToString('o') }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri "$base/api/tickets/$($ticket.id)/messages" -ContentType 'application/json' -Body $body
Invoke-RestMethod "$base/api/tickets/$($ticket.id)/risk-history"
```

Interactive OpenAPI documentation is at [localhost:8000/docs](http://localhost:8000/docs).

## Reproduce and verify

- [Windows setup and retraining](docs/WINDOWS_SETUP.md)
- [Demo and screenshot walkthrough](docs/DEMO.md)
- [Architecture and storage](docs/ARCHITECTURE.md)
- [Features, calibration and thresholds](docs/MODELING.md)
- [Evaluation protocol, results and failure analysis](docs/EVALUATION.md)
- [Machine-readable evaluation results](artifacts/evaluation-results.json)

For local verification with Python 3.12 and Node 22, install `requirements-dev.txt`, run `python -m pytest -q`, then run `npm ci`, `npm test` and `npm run build` from `frontend`. Setting `TEST_POSTGRES_URL` enables PostgreSQL integration tests against a dedicated test database. Hosted CI also exercises Docker Compose startup.

The system is advisory and designed for a local portfolio demo. Synthetic outcomes and a small wording bank limit external validity. The service has no real ticket lifecycle integration or operator authentication. It treats all created tickets as active.

### Build verification

The delivered experiment trained and evaluated all three candidates and the ablations. The backend passed 12 tests with SQLite and PostgreSQL 18.7, with the SQLite instance of the PostgreSQL-only lock test skipped. Three frontend tests passed, including replay through a running Vite proxy, FastAPI and PostgreSQL. The production frontend built successfully. The development environment had no Docker daemon and blocked graphical Chromium initialization. Docker Desktop/Compose startup and a full browser rendering remain unverified here.
