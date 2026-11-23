# Windows setup

Install Docker Desktop with the WSL 2 backend and start it. Extract the ZIP, then open the inner `EscalationIQ` folder in VS Code. Open a PowerShell terminal there.

```powershell
docker compose version
docker compose up --build -d --wait
Invoke-RestMethod http://localhost:8000/health
Start-Process http://localhost:5173
```

The ZIP includes the deployed model and reports. The initial Docker build needs internet for packages and container images. It does not need an API key, local Python, local Node or a separately installed PostgreSQL server.

To populate the queue immediately, run the demo seeder and click Refresh.

```powershell
docker compose exec api python -m escalation.seed
```

Every seed run creates fresh sample ticket IDs. Stop services with `docker compose down`. PostgreSQL data is kept in the named volume between runs.

## Reproduce training

```powershell
docker compose --profile training run --build --rm train --seed 42
Get-Content artifacts/model-info.json
docker compose up --build -d --wait
```

The training image installs CPU PyTorch and downloads the pinned [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model, about 90 MB of weights plus tokenizer/configuration. The Hugging Face cache uses a named volume. Later training runs reuse the cache. The bundled deployment uses the baseline and does not load the encoder. If a different training run selects hybrid, set `$env:INSTALL_ENCODER = '1'` in the same terminal before rebuilding. If the encoder cache is empty, the first semantic request needs network access.

Set optional demo settings by copying `.env.example` to `.env` and editing it. Set a database password before initializing the PostgreSQL volume if you want to change the local default. The application receives local requests on ports 5173 and 8000.

## Troubleshooting

- If Compose cannot reach Docker, start Docker Desktop and wait for its engine to finish starting.
- If either host port is in use, stop the application holding it or change the corresponding host port in `compose.yaml`.
- If health stays pending, run `docker compose ps` and `docker compose logs api db`. The API starts after PostgreSQL and applies migrations.
- If replay fails after a transient network error, use **Replay next message** again. The browser reuses that message UUID.
