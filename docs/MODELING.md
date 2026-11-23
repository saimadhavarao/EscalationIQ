# Modeling

A score estimates whether a ticket will eventually reach a simulated escalation event given the currently observed prefix. The target does not specify a fixed future time window. Each training prefix receives the ticket outcome for supervision. Positive prefixes stop strictly before the event, and negative prefixes extend through the last observed message.

## Feature definitions

| Group | Inputs |
|---|---|
| Volume | Prefix message count and customer messages divided by agent messages with a denominator floor of one |
| Questions and repetition | Question marks since the last agent reply and lexical overlap against up to four earlier customer messages |
| Timing | Elapsed duration, average first-pending-customer-to-agent reply delay, and minutes since last agent reply or conversation start |
| Troubleshooting | Agent messages with step terms and repeated requests immediately after a detected suggestion |
| Tone | Small lexicon count of negative minus positive terms, most recent customer tone, and last-two tone mean relative to the opening customer turn |
| History and category | Creation-time category, simulated prior ticket count, prior escalations and tenure with missing indicators; missing numerics use zero |
| TF-IDF | Word unigram/bigram TF-IDF over the most recent five messages with role markers |
| Semantic | 384-d MiniLM pooled and last-turn embeddings, adjacent cosine similarity, recent customer repetition cosine and mean customer-to-agent cosine distance |

Open questions and unsuccessful attempts are proxies. The trend lexicon has limited vocabulary and can misread negation. Cosine distance can capture topic change and does not establish that an agent failed to answer. Historical counts and tenure are generated, and the live service accepts only caller-supplied creation-time values.

## Estimators and selection

- Baseline fits class-balanced logistic regression over standardized structured features and training-fit TF-IDF. It searches C in {0.1, 0.5}.
- Structured fits class-balanced histogram gradient boosting. It searches leaf limits in {7, 15}.
- Hybrid fits class-balanced logistic regression over structured and semantic features. It searches C in {0.03, 0.12}.
- Ablations remove embeddings, tone trend or simulated historical fields and their missingness flags. The separate boosting candidate is the structured-only nonlinear comparison.

Prefix weights sum to one per ticket. Each candidate receives sigmoid calibration on the first 90 validation tickets. The remaining 90 choose the high threshold for maximum ticket-level F1 with at most 25% alerts. Candidate and model selection use validation F1, then average precision, then lower Brier as tie-breakers. Only the three primary models are eligible for deployment.

Low and medium are advisory bands split at 65% of the high cutoff. Critical is 35% of the way from high to 1. These severity bands are operational heuristics. Same or lower severity never repeats an alert on a ticket, and high can upgrade to critical after the configured cooldown. A 10 percentage-point increase is recorded as a rapid rise. Only the primary high cutoff is tuned under the offline alert budget.

Observed signals describe measurable properties of the transcript. Model sensitivity replaces a single structured feature with its training median, holds other features fixed, and reports the calibrated risk difference. Correlated features may make those counterfactual inputs unrealistic. The displayed deltas do not add up to the score, make causal claims, or attribute embeddings to tokens.
