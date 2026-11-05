# Latent Information Dynamics — Methodology Freeze

Freeze date: 2026-11-06

Git checkpoint immediately before freeze:

35140c1 add frozen 24h neural robustness validation

## Purpose

This document freezes the empirical methodology before any evaluation
of the 2026 confirmation sample.

The 2026 sample is confirmation only. It must not be used for model
selection, architecture changes, optimizer changes, feature selection,
hyperparameter tuning, modality gating, or post-hoc intervention design.

## Primary Research Question

Does a multimodal latent representation containing point-in-time Price
and price-independent trade-flow information improve prediction of
future Bitcoin returns beyond a matched Price-only representation?

## Development Sample

Development decisions:

2020-01-02 through 2025-12-31.

All methodological development was completed on this period.

## Confirmation Sample

Planned confirmation decisions:

2026-01-02 through 2026-06-30.

The exact usable end date will depend on available feature and target
resolution data.

## Primary Horizon

Primary forecast horizon:

4 hours.

The 4h specification remains the primary experiment.

## Robustness Horizon

Secondary robustness horizon:

24 hours.

No model or optimizer changes were made in response to the 24h result.

## Frozen Features

Price modality — 11 features:

- return_5m
- return_30m
- return_1h
- return_4h
- return_12h
- return_24h
- rv_1h
- rv_4h
- rv_24h
- range_4h
- range_24h

Flow modality — 20 price-independent features:

For each of:

- 5m
- 15m
- 30m
- 1h
- 4h

the following are used:

- total_quantity
- signed_quantity
- quantity_imbalance
- trade_count

## Frozen Models

### M0

Historical unconditional Bernoulli probability calculated only from
labels legally available at the prediction boundary.

### Price NN

Frozen PriceLatentModel.

Architecture:

11 -> 16 -> 4
4 -> 8 -> 4
4 -> binary logit

Activation:

SiLU

### Multimodal NN

Frozen MultimodalLatentModel.

Price encoder:

11 -> 16 -> 4

Flow encoder:

20 -> 24 -> 4

Fusion:

8 -> 8 -> 4

Prediction head:

4 -> binary logit

Activation:

SiLU

## Frozen Training Protocol

Optimizer:

AdamW

Learning rate:

0.001

Weight decay:

0.0001

Batch size:

128

Maximum epochs:

50

Base seed:

20261016

Execution:

CPU with deterministic PyTorch algorithms.

## Epoch Selection

For every outer prediction boundary:

1. Construct the complete legally available training sample.
2. Reserve the final 546 chronological observations as validation.
3. Require all inner-training labels to be available by validation start.
4. Train candidate epochs from 1 through 50.
5. Select the epoch with minimum validation log loss.
6. Discard that selection model.
7. Refit from scratch on the complete legal outer training sample.
8. Train for exactly the selected number of epochs.
9. Produce OOS forecasts.

This procedure is frozen.

## Point-in-Time Requirement

For every training observation:

label_available_time <= prediction boundary.

No future feature information or unresolved labels may enter training.

## Primary Metrics

Primary:

Log loss.

Secondary:

Brier score.

Descriptive:

Classification accuracy.

Model comparisons use observation-level loss differences with HAC
standard errors.

Frozen HAC lags:

0, 1, 6, 12, 42.

## Development Findings

### 4h Neural Experiment

Price NN:

log loss = 0.69051543

Multimodal NN:

log loss = 0.69286179

Multimodal minus Price NN:

+0.00234636

Inference was borderline and HAC-lag sensitive.

Interpretation:

Price information was comparatively stable.

Flow information was useful in many folds but occasionally produced
large OOS deterioration.

This is not interpreted as proof that Flow contains no incremental
information.

## Modality Intervention

Flow-neutral predictions numerically repaired much of the aggregate
multimodal gap, especially in several extreme failure folds.

However, Flow was helpful in 12 of 20 folds and harmful in 8.

Permanent interpretation:

Flow appears conditionally useful but exhibits asymmetric failure risk
under some regimes.

## Gradient Interference

Counterfactual Price and Flow shared-gradient cosine became more
conflicted with longer training.

Greater disagreement was associated with worse multimodal OOS
performance.

This is treated as an exploratory diagnostic and not as an exact
additive decomposition of the optimization gradient.

## Rejected Interventions

The following were investigated and rejected as final methodology:

- gradient clipping
- simple conflict damping
- fixed recency-window training
- feature-space drift gating
- early-OOS drift gating
- coordinatewise median-of-means gradient estimation

No further BTC intervention tuning is permitted before confirmation.

## Synthetic Evidence

Controlled synthetic experiments established that the multimodal
architecture can exploit useful Flow information under stationary
conditions.

Heavy-tailed Flow reduced that advantage.

Flow scale shifts materially harmed multimodal performance.

Flow sign reversal caused severe failure.

Static optimizer changes did not solve semantic modality reversal.

AdamW remains the frozen optimizer.

## 24h Robustness Result

OOS forecasts:

10,830

M0:

log loss = 0.69532996

Price NN:

log loss = 0.69628346

Price NN minus M0:

+0.00095351

Not statistically significant across the frozen HAC lags.

Multimodal NN:

log loss = 0.69553434

Multimodal NN minus Price NN:

-0.00074912

Not statistically significant across the frozen HAC lags.

Interpretation:

The predictive structure observed at 4h does not robustly extrapolate
to the 24h target.

Neither learned model clearly beats the historical unconditional
benchmark at 24h.

No tuning will be performed in response.

## Final 2026 Confirmation Questions

The primary 2026 questions are:

1. Does the frozen 4h Price NN retain predictive value relative to M0?

2. Does the frozen Multimodal NN improve or deteriorate performance
   relative to the frozen Price NN?

The 2026 result will be reported regardless of outcome.

Any method developed after observing 2026 predictive performance must
be explicitly labelled post-confirmation research and cannot be part of
the frozen confirmation experiment.
