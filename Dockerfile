FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HF_HOME=/cache/huggingface
COPY requirements.txt requirements-ml.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

FROM base AS training
RUN python -m pip install --no-cache-dir torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements-ml.txt
COPY escalation ./escalation
ENTRYPOINT ["python", "-m", "escalation.train"]

FROM base AS runtime
ARG INSTALL_ENCODER=0
RUN if [ "$INSTALL_ENCODER" = "1" ]; then \
      python -m pip install --no-cache-dir torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu \
      && python -m pip install --no-cache-dir -r requirements-ml.txt; fi
COPY escalation ./escalation
COPY artifacts ./artifacts
COPY migrations ./migrations
COPY alembic.ini ./
EXPOSE 8000
CMD ["sh", "-c", "python -m alembic upgrade head && python -m uvicorn escalation.api:create_app --factory --host 0.0.0.0 --port 8000"]
