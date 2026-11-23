# Evaluation

## Data provenance

The [Taskmaster-1 dataset](https://github.com/google-research-datasets/Taskmaster/blob/master/TM-1-2019/README.md) is licensed CC BY 4.0. It labels task dialogue with API arguments and transaction status. Transaction rejection is not the same event as a specialist escalation. [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) exposes public reply links and timestamps, but inspecting the published schema did not establish reliable off-platform escalations or a lifecycle label. The [Monodox dataset](https://huggingface.co/datasets/monodox/customer-support-and-service) provides synthetic support examples and escalation flags, and its published record schema is not a message-by-message event history. No external conversation text was copied into the repository.

The project-generated corpus uses seed 42 with 480 training tickets, 180 validation tickets and 240 holdout tickets. The holdout has 77 escalations (32.1%). All tickets are fictional. Friction, urgency, prior workload and simulated unresolved issues affect a stochastic escalation hazard. The generator varies conversation length, ordering, delays and phrasing. Routine cases have a lower hazard. Historical fields are synthetic, sometimes missing, and stored as creation-time metadata. Negative tickets include routine and non-routine conversations that end without a simulated escalation. The simulator does not model a complete service resolution taxonomy.

## Leakage safeguards

| Risk | Safeguard |
|---|---|
| Future messages | Offline features receive slices ending at the prediction position. The live transaction retrieves only previously committed messages and appends the current one. Tests mutate future text and outcomes and compare the same prefix. |
| Final outcome | The event position is available only to the sampler and evaluator. The API's message and context schemas reject extra fields. Positive examples stop before their event message. |
| Escalation-label leakage | No label, event timing, final ticket state, ticket identifier, or template identifier enters feature extraction. |
| Duplicate tickets or customers | The generator deduplicates exact role/text transcripts and audits identifiers, customers and transcripts across complete splits. |
| Template overlap | Cohorts have distinct phrasing families with automated utterance-disjointness checks. Categories and the latent simulation mechanism remain shared. |
| Test-fit preprocessing | The classifier, TF-IDF vocabulary, scaler and reference medians are fit on training only. The pretrained encoder is frozen. Calibration uses the earlier validation partition. |
| Holdout threshold search | Model choice and thresholds are frozen from later validation before test data is scored. The retrospective top-budget ranking is a descriptive metric. |

## Protocol

A ticket alerts if any eligible prefix crosses the high threshold. Positive tickets use only messages strictly before escalation. Negative tickets use all observed messages. Precision and recall at fixed budgets rank by the maximum eligible score, with stable cohort ordering for ties. This retrospective ranking is distinct from the chronological live alert policy.

The confusion matrix is ordered as actual negative/positive by predicted no-alert/alert. Lead time uses the first eligible alert. Lead-time medians use detected tickets only. Detection coverage at one, two and three messages divides by all 77 escalated tickets, including never-alerted tickets. Confidence intervals bootstrap complete tickets with frozen model and threshold over 350 replicates. Prefix Brier and binned calibration error each give equal total weight to each ticket. Warm latency includes prefix features, model scoring and structured sensitivity. It excludes database and HTTP time. Hybrid timing reuses cached message embeddings.

## Results

| Model / ablation | Precision | Recall | F1 | Average precision | ROC-AUC | Prefix Brier | False alerts / 100 | Warm p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.400 | 0.156 | 0.224 | 0.371 | 0.565 | 0.204 | 7.500 | 2.74 |
| structured | 0.500 | 0.208 | 0.294 | 0.423 | 0.597 | 0.204 | 6.667 | 1.95 |
| hybrid | 0.375 | 0.195 | 0.256 | 0.358 | 0.573 | 0.210 | 10.417 | 0.93 |
| without embeddings | 0.408 | 0.260 | 0.317 | 0.420 | 0.590 | 0.198 | 12.083 | 0.74 |
| without sentiment trend | 0.385 | 0.195 | 0.259 | 0.358 | 0.574 | 0.210 | 10.000 | 0.96 |
| without history | 0.308 | 0.208 | 0.248 | 0.352 | 0.572 | 0.210 | 15.000 | 0.92 |

The deployed cutoff was **0.5614**. It alerted on 30/240 tickets. The confusion matrix was `[[145, 18], [65, 12]]`. Prefix ECE was 0.052. The ticket-bootstrap 95% intervals were 0.210–0.567 for precision, 0.071–0.253 for recall, and 0.281–0.476 for average precision.

At least one, two and three messages ahead, it detected **12/77**, **12/77** and **9/77** respectively. The 12 detections had median lead of four messages and 29 minutes. At a retrospective 25% alert budget, precision was 38.3% and recall was 29.9%.

## Failure analysis

The baseline missed 65 of 77 escalations. Billing accounted for 12 misses and technical cases for 10. Refund and access contributed 11 of 18 false alerts. The later wording cohort induced a substantial validation-to-holdout F1 drop. The structured model ranked holdout tickets better, and the linear structured ablation outscored the hybrid in holdout F1. The holdout was not used to switch deployment. This simulator shares a hazard function and category vocabulary across cohorts, so these results only quantify performance under the implemented distribution shift. Real-world results would need independently labeled escalation timestamps and a fresh chronological test.

See the [JSON report](../artifacts/evaluation-results.json) for per-model matrices, intervals, first-alert lead times, calibration bins, validation scores and fixed-budget metrics.
