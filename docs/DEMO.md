# Demo walkthrough

1. Start the services using the Windows setup instructions and open [the dashboard](http://localhost:5173).
2. Open **Live replay**. Keep the first scenario selected and click **Replay next message** until the conversation ends. The first three scenarios are independent simulated cases curated for high peak risk. The last three illustrate quieter cases.
3. Inspect the plotted score after each message. Read the recommendation, risk change, first alert position, and any alert suppression decision. Compare **Observed signals** with **Model sensitivity**.
4. Open **Monitor**. The replay now appears in the ticket queue. Filter or sort the queue, click a ticket ID, and compare its persisted history with the replay. Use **Refresh** after creating tickets in another window.
5. Open **Evaluation**. Compare candidate models and ablations, then review lead time, confusion matrix and alert-budget curves. The offline holdout remains the same while you replay tickets.
6. To produce screenshots, capture **Live replay** after the first alert and **Evaluation** after comparing the candidates. Include the synthetic holdout label when sharing the latter.

The simulator submits one real API request per message. It never submits a final disposition or escalation label. The sample picker was curated for demonstration, so use the chronological holdout for performance claims.
