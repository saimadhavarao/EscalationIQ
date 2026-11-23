from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Policy:
    high: float
    cooldown_minutes: int = 45

    @property
    def critical(self):
        return self.high + (1 - self.high) * 0.35

    def decide(self, risk: float, at: datetime, previous: dict | None,
               previous_alert: tuple[str, datetime] | None) -> dict:
        if risk >= self.critical:
            level, action = "critical", "Recommend immediate specialist review"
        elif risk >= self.high:
            level, action = "high", "Recommend human specialist review"
        elif risk >= self.high * 0.65:
            level, action = "medium", "Monitor the conversation"
        else:
            level, action = "low", "Continue normal support"
        delta = risk - previous["risk"] if previous else 0.0
        alert = level in ("high", "critical")
        reason = "Threshold crossed" if alert else "Below alert threshold"
        if alert and previous_alert:
            old_level, old_time = previous_alert
            if old_level == "critical" or level == old_level:
                alert, reason = False, "Already alerted at this severity or higher"
            elif (at - old_time).total_seconds() < self.cooldown_minutes * 60:
                alert, reason = False, "Cooldown active"
            else:
                reason = "Severity increased after cooldown"
        return {"level": level, "action": action, "delta": delta, "rapid_increase": delta >= 0.10,
                "alert": alert, "alert_reason": reason}
