import argparse
import hashlib
import json
from time import perf_counter
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .data import generate
from .evaluate import operating_threshold, summarize
from .features import EMBEDDING, REVISION, HISTORY, warm_embeddings
from .model import RiskModel, row


def prefixes(ticket):
    end = ticket.event if ticket.event is not None else len(ticket.messages) + 1
    return [ticket.messages[:i] for i in range(1, end)]


def examples(tickets, embeddings):
    return [[row(prefix, t.context, embeddings) for prefix in prefixes(t)] for t in tickets]


def flatten(tickets, groups):
    sizes = [len(group) for group in groups]
    return [item for group in groups for item in group], np.repeat(
        [t.event is not None for t in tickets], sizes
    ), np.concatenate([np.full(size, 1 / size) for size in sizes])


def trajectories(model, groups):
    lengths = np.cumsum([len(g) for g in groups])
    return np.split(model.probabilities([r for g in groups for r in g]), lengths[:-1])


def run(output, seed, structured_only=False, counts=(480, 180, 240)):
    splits = generate(seed, counts)
    train, validation, test = (splits[s] for s in ("train", "validation", "test"))
    cutoff = len(validation) // 2
    calibration, tuning = validation[:cutoff], validation[cutoff:]
    if not structured_only:
        warm_embeddings({m.text for t in train + validation for p in prefixes(t) for m in p})
    training_rows = examples(train, not structured_only)
    calibration_rows = examples(calibration, not structured_only)
    tuning_rows = examples(tuning, not structured_only)
    candidates = {
        "baseline": ("baseline", (0.1, 0.5), ()),
        "structured": ("structured", (7, 15), ()),
    }
    if not structured_only:
        candidates.update({
            "hybrid": ("hybrid", (0.03, 0.12), ()),
            "without_embeddings": ("linear_structured", (0.03, 0.12), ()),
            "without_sentiment_trend": ("hybrid", (0.03, 0.12), ("tone_trend",)),
            "without_history": ("hybrid", (0.03, 0.12),
                                tuple(c for k in HISTORY for c in (k, f"{k}_missing"))),
        })
    fitted = {}
    validation_report = {}
    for name, (kind, strengths, dropped) in candidates.items():
        best = None
        for strength in strengths:
            model = RiskModel(kind, strength, dropped).fit(*flatten(train, training_rows))
            model.calibrate(*flatten(calibration, calibration_rows))
            scores = trajectories(model, tuning_rows)
            threshold = operating_threshold(tuning, scores)
            metrics = summarize(tuning, scores, threshold)
            selection = (metrics["f1"], metrics["pr_auc"], -metrics["prefix_brier"])
            if best is None or selection > best[0]:
                best = (selection, model, threshold, metrics)
        fitted[name] = best
        validation_report[name] = {"strength": best[1].strength, **best[3]}
        print(name, "validation F1", round(best[3]["f1"], 3), flush=True)

    selected = max((name for name in fitted if name in ("baseline", "structured", "hybrid")),
                   key=lambda name: fitted[name][0])
    _, chosen, threshold, _ = fitted[selected]
    output.mkdir(exist_ok=True, parents=True)
    artifact_path = output / "model.joblib"
    info = {"selected": selected, "seed": seed, "threshold": threshold,
            "sklearn": sklearn.__version__, "encoder": EMBEDDING if not structured_only else None,
            "encoder_revision": REVISION if not structured_only else None}
    joblib.dump({"model": chosen, "info": info}, artifact_path, compress=3)
    info["version"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:12]
    joblib.dump({"model": chosen, "info": info}, artifact_path, compress=3)

    # The selected model, hyperparameters and thresholds are frozen here.
    if not structured_only:
        warm_embeddings({m.text for t in test for p in prefixes(t) for m in p})
    test_rows = examples(test, not structured_only)
    holdout = {}
    for name, (_, model, threshold, _) in fitted.items():
        holdout[name] = summarize(test, trajectories(model, test_rows), threshold, bootstrap=350)
        latencies = []
        for ticket in test[:24]:
            for prefix in prefixes(ticket):
                start = perf_counter()
                current = row(prefix, ticket.context, model.kind == "hybrid")
                risk = model.probabilities([current])[0]
                model.explain(current, risk)
                latencies.append((perf_counter() - start) * 1000)
        holdout[name]["inference_latency_ms"] = dict(zip(("p50", "p95"), np.quantile(latencies, [0.5, 0.95])))
    result = {"model": info, "split_counts": dict(zip(splits, counts)), "validation": validation_report,
              "holdout": holdout, "inference_latency_ms": holdout[selected]["inference_latency_ms"]}
    (output / "evaluation-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "model-info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Curated replay uses an independent seed and is excluded from all metrics.
    demo = []
    demo_pool = generate(seed + 1, (0, 0, 120))["test"]
    demo_rows = examples(demo_pool, chosen.kind == "hybrid")
    demo_scores = trajectories(chosen, demo_rows)
    for event in (True, False):
        ranked = sorted(((t, max(scores)) for t, scores in zip(demo_pool, demo_scores)
                         if (t.event is not None) == event),
                        key=lambda item: item[1], reverse=event)
        for ticket, _ in ranked[:3]:
            demo.append({"name": f"{ticket.context.category.title()} · case {ticket.id}",
                         "context": ticket.context.model_dump(),
                         "messages": [m.model_dump(mode="json") for m in prefixes(ticket)[-1]]})
    (output / "demo.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
    print("Selected", selected, "holdout", {k: holdout[selected][k] for k in ("precision", "recall", "pr_auc")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--without-encoder", action="store_true")
    parser.add_argument("--counts", nargs=3, type=int, default=(480, 180, 240))
    args = parser.parse_args()
    run(args.output, args.seed, args.without_encoder, tuple(args.counts))
