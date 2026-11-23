"""Ticket-level early warning evaluation and paired bootstrap."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, confusion_matrix

ALERT_BUDGET = 0.25


def summarize(tickets, trajectories, threshold, bootstrap=0, seed=42):
    labels = np.array([t.event is not None for t in tickets], dtype=int)
    peaks = np.array([max(scores) for scores in trajectories])
    alerts = peaks >= threshold
    cm = confusion_matrix(labels, alerts, labels=[0, 1]).tolist()
    tn, fp, fn, tp = np.asarray(cm).ravel().tolist()
    leads, minutes = [], []
    for ticket, scores in zip(tickets, trajectories, strict=True):
        if ticket.event is not None and max(scores) >= threshold:
            first = next(i for i, score in enumerate(scores, 1) if score >= threshold)
            leads.append(ticket.event - first)
            minutes.append((ticket.messages[ticket.event - 1].at - ticket.messages[first - 1].at)
                           .total_seconds() / 60)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    total = len(tickets)
    curve = []
    for fraction in (0.05, 0.10, 0.15, ALERT_BUDGET, 0.35, 0.5):
        budget = max(1, int(fraction * total))
        top = np.argsort(-peaks, kind="stable")[:budget]
        curve.append({"budget": fraction, "precision": float(labels[top].mean()),
                      "recall": float(labels[top].sum() / max(1, labels.sum()))})
    # Brier uses the calibrated per-prefix prediction, weighting tickets equally.
    weights = np.concatenate([np.full(len(scores), 1 / len(scores)) for scores in trajectories])
    prefix_labels = np.repeat(labels, [len(scores) for scores in trajectories])
    prefix_prob = np.concatenate(trajectories)
    reliability = []
    for lower, upper in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (prefix_prob >= lower) & ((prefix_prob < upper) if upper < 1 else (prefix_prob <= upper))
        if mask.any():
            reliability.append({"lower": float(lower), "upper": float(upper),
                                "predicted": float(np.average(prefix_prob[mask], weights=weights[mask])),
                                "observed": float(np.average(prefix_labels[mask], weights=weights[mask])),
                                "weight": float(weights[mask].sum() / weights.sum())})
    categories = {}
    for ticket, alerted in zip(tickets, alerts, strict=True):
        category = categories.setdefault(ticket.context.category, {"tickets": 0, "missed": 0, "false_alerts": 0})
        category["tickets"] += 1
        category["missed"] += int(ticket.event is not None and not alerted)
        category["false_alerts"] += int(ticket.event is None and alerted)
    result = {
        "tickets": total, "positives": tp + fn, "threshold": threshold,
        "precision": precision, "recall": recall, "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "pr_auc": float(average_precision_score(labels, peaks)),
        "roc_auc": float(roc_auc_score(labels, peaks)),
        "prefix_brier": float(brier_score_loss(prefix_labels, np.concatenate(trajectories), sample_weight=weights)),
        "prefix_ece": sum(b["weight"] * abs(b["predicted"] - b["observed"]) for b in reliability),
        "reliability_bins": reliability,
        "by_category": categories,
        "confusion_matrix": cm, "alert_rate": (tp + fp) / total,
        "false_alerts_per_100": 100 * fp / total, "alert_budget": curve,
        "lead_messages": leads, "lead_minutes": minutes,
        "detected_at_least": {str(k): sum(lead >= k for lead in leads) for k in (1, 2, 3)},
    }
    if bootstrap:
        rng = np.random.default_rng(seed)
        resamples = []
        for _ in range(bootstrap):
            idx = rng.integers(0, total, size=total)
            if 0 < labels[idx].sum() < total:
                y, a = labels[idx], alerts[idx]
                hits = (y * a).sum()
                resamples.append([average_precision_score(y, peaks[idx]),
                                  hits / max(1, a.sum()), hits / y.sum()])
        result["ci95"] = {key: np.quantile(values, [0.025, 0.975]).tolist()
                          for key, values in zip(("pr_auc", "precision", "recall"), np.asarray(resamples).T)}
    return result


def operating_threshold(tickets, trajectories):
    peaks = np.array([max(scores) for scores in trajectories])
    labels = np.array([t.event is not None for t in tickets])
    max_alerts = int(ALERT_BUDGET * len(tickets))
    feasible = []
    for threshold in [*np.unique(peaks), 1.0]:
        alerts = peaks >= threshold
        if alerts.sum() <= max_alerts:
            tp = (alerts & labels).sum()
            f1 = 2 * tp / max(1, alerts.sum() + labels.sum())
            feasible.append((f1, tp, float(threshold)))
    return max(feasible)[2]
