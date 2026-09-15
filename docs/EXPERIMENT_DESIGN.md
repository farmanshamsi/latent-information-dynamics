# Experiment Design

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

This document defines the empirical testing framework for the research programme.

The purpose is to determine whether information observable before Bitcoin price adjustment contains predictive information beyond Bitcoin's own historical price.

The experiment is designed around the comparison

$$
\mathcal{M}_{P}
\quad\text{vs.}\quad
\mathcal{M}_{I}
\quad\text{vs.}\quad
\mathcal{M}_{I+P}.
$$

The central principle is:

> Model complexity is secondary. Incremental out-of-sample information is the object being tested.

---

# 1. Primary Research Question

Does non-price information available at time $t$ improve the prediction of the Bitcoin return over horizon $h$ beyond information already contained in Bitcoin's historical price process?

Formally,

$$
H_0:
\mathcal{L}
\left(
\mathcal{M}_{I+P}
\right)
\geq
\mathcal{L}
\left(
\mathcal{M}_{P}
\right),
$$

against

$$
H_1:
\mathcal{L}
\left(
\mathcal{M}_{I+P}
\right)
<
\mathcal{L}
\left(
\mathcal{M}_{P}
\right),
$$

where $\mathcal{L}$ denotes out-of-sample predictive loss.

A stronger result would establish

$$
\mathcal{L}
\left(
\mathcal{M}_{I}
\right)
<
\mathcal{L}
\left(
\mathcal{M}_{P}
\right).
$$

---

# 2. Prediction Target

For Bitcoin reference price $S_t$, define

$$
R_{t,h}
=
\log
\left(
\frac{S_{t+h}}{S_t}
\right).
$$

The initial binary target is

$$
Y_{t,h}
=
\mathbf{1}
\left\{
R_{t,h}>\tau_h
\right\}.
$$

Two threshold specifications will be considered during development.

## Zero threshold

$$
\tau_h=0.
$$

This predicts pure direction.

## Economic threshold

$$
\tau_h
=
c_h,
$$

where $c_h$ reflects an estimate of noise and transaction costs.

The final threshold specification will be frozen before final holdout evaluation.

---

# 3. Forecast Horizons

Candidate horizons are initially

$$
\mathcal{H}
=
\{
1m,
5m,
15m,
1h,
4h,
1d
\}.
$$

Not every horizon will necessarily remain in the final paper.

The development stage will determine a small number of primary horizons representing distinct information-transmission speeds.

A likely structure is:

- short horizon: microstructure dominated,
- medium horizon: multimodal information assimilation,
- longer horizon: slower macro, news and on-chain transmission.

The final primary horizons must be fixed before opening the locked holdout period.

---

# 4. Model Families

The experiment follows a complexity ladder.

A more complex model will not replace a simpler benchmark merely because it is more sophisticated.

---

# 5. Benchmark 0 — Unconditional Probability

The simplest benchmark estimates

$$
\hat p
=
\frac{1}{N}
\sum_{t=1}^{N}Y_t.
$$

Every evaluation observation receives the same estimated probability.

This establishes the minimum probabilistic benchmark.

---

# 6. Benchmark 1 — Price-Only Model

The principal benchmark is

$$
\mathcal{M}_{P}.
$$

Candidate price features may include:

- lagged returns,
- realized volatility,
- range measures,
- momentum,
- recent drawdown,
- price acceleration,
- volume where appropriate.

The baseline feature vector is

$$
\mathbf{p}_t
=
[
R_{t-1},
R_{t-2},
\ldots,
RV_t,
Momentum_t,
\ldots
]^\top.
$$

At least one simple price-only model must remain throughout the entire research process.

---

# 7. Price-Only Model Ladder

The initial price model ladder is:

1. logistic regression,
2. regularized logistic regression,
3. random forest,
4. gradient boosting.

Only after these baselines are established may more complex temporal models be introduced.

Possible later extensions include:

- LSTM,
- temporal convolution,
- attention-based sequence models.

The purpose of these models is to establish a strong benchmark, not to make price history the central research contribution.

---

# 8. Information-Only Models

The information-only system is

$$
\mathcal{M}_{I}.
$$

The initial information domains are:

$$
\mathbf{i}_t
=
[
\mathbf{o}_t,
\mathbf{n}_t,
\mathbf{c}_t,
\mathbf{x}_t,
\mathbf{v}_t
].
$$

Separate experiments will first evaluate each domain.

---

# 9. Microstructure Model

The microstructure experiment estimates

$$
p_{t,h}^{O}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{o}_t
\right).
$$

Candidate variables include:

- order-book imbalance,
- trade imbalance,
- order-flow imbalance,
- spread,
- depth,
- cancellation intensity,
- replenishment,
- queue depletion,
- short-horizon price impact,
- cross-venue dislocation.

This is expected to be especially relevant at shorter horizons.

---

# 10. News Model

The news experiment estimates

$$
p_{t,h}^{N}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{n}_t
\right).
$$

Candidate variables include:

- sentiment,
- event type,
- entity,
- novelty,
- uncertainty,
- relevance,
- time since publication,
- event persistence.

News experiments must use publication or availability time rather than article date alone.

---

# 11. On-Chain Model

The blockchain information model estimates

$$
p_{t,h}^{C}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{c}_t
\right).
$$

Potential variables include:

- exchange flows,
- miner flows,
- network activity,
- mempool state,
- fees,
- hash rate,
- mining difficulty,
- stablecoin liquidity,
- large-holder activity.

Each variable must pass the point-in-time availability audit.

---

# 12. Cross-Market Model

The cross-market model estimates

$$
p_{t,h}^{X}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{x}_t
\right).
$$

Potential features include:

- cross-asset returns,
- lead-lag variables,
- spread variables,
- cointegration residuals,
- error-correction terms,
- volatility state,
- market-open indicators,
- stale-price age.

---

# 13. Derivatives and Regime Model

A secondary state model may estimate

$$
p_{t,h}^{V}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{v}_t
\right).
$$

Candidate state variables include:

- implied volatility,
- funding,
- futures basis,
- open interest,
- liquidation intensity,
- volatility regime.

This layer is secondary unless empirical results justify its inclusion.

---

# 14. Combined Information Model

The full information-only system is

$$
p_{t,h}^{I}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{o}_t,
\mathbf{n}_t,
\mathbf{c}_t,
\mathbf{x}_t,
\mathbf{v}_t
\right).
$$

Two fusion strategies will be compared.

## Early fusion

Concatenate feature vectors:

$$
\mathbf{i}_t
=
[
\mathbf{o}_t^\top,
\mathbf{n}_t^\top,
\mathbf{c}_t^\top,
\mathbf{x}_t^\top,
\mathbf{v}_t^\top
]^\top.
$$

Then estimate

$$
p_t
=
f(\mathbf{i}_t).
$$

## Late fusion

Each specialist model produces

$$
p_t^{(j)}.
$$

A second-stage model combines

$$
p_t
=
g
\left(
p_t^{O},
p_t^{N},
p_t^{C},
p_t^{X},
p_t^{V}
\right).
$$

Late fusion is attractive because the information domains have different structures, frequencies and statistical properties.

---

# 15. Combined Price and Information Model

The broadest predictive system is

$$
p_{t,h}^{I+P}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,
\mathbf{p}_t
\right).
$$

This is the most important model for the central incremental-information test.

If

$$
\mathcal{M}_{I+P}
$$

reliably outperforms

$$
\mathcal{M}_{P},
$$

then non-price information contributes predictive information beyond historical price.

---

# 16. Experiment Ladder

The empirical sequence is deliberately cumulative.

### Experiment A

$$
\text{Unconditional}
\rightarrow
\mathcal{M}_{P}.
$$

Question:

Does historical price contain forecastable directional structure?

### Experiment B

$$
\mathcal{M}_{P}
\quad\text{vs.}\quad
\mathcal{M}_{O}.
$$

Question:

Does microstructure outperform price history at short horizons?

### Experiment C

$$
\mathcal{M}_{P}
\quad\text{vs.}\quad
\mathcal{M}_{O}
\quad\text{vs.}\quad
\mathcal{M}_{P+O}.
$$

Question:

Does microstructure add incremental information?

### Experiment D

Introduce one additional information domain at a time.

### Experiment E

Construct

$$
\mathcal{M}_{I}.
$$

### Experiment F

Construct

$$
\mathcal{M}_{I+P}.
$$

### Experiment G

Evaluate latent-state and regime-dependent extensions only if earlier experiments justify additional complexity.

---

# 17. Ablation Design

Once the multimodal model exists, each information block will be removed separately.

Let

$$
\mathcal{M}_{full}
$$

be the complete model.

For domain $j$, define

$$
\mathcal{M}_{-j}
$$

as the model without that domain.

The incremental contribution is

$$
\Delta_j
=
\mathcal{L}(\mathcal{M}_{-j})
-
\mathcal{L}(\mathcal{M}_{full}).
$$

If

$$
\Delta_j>0,
$$

removing domain $j$ worsens predictive performance.

---

# 18. Pairwise Incremental Tests

Important comparisons include

$$
\mathcal{M}_{P+O}
\quad\text{vs.}\quad
\mathcal{M}_{P},
$$

$$
\mathcal{M}_{P+N}
\quad\text{vs.}\quad
\mathcal{M}_{P},
$$

$$
\mathcal{M}_{P+C}
\quad\text{vs.}\quad
\mathcal{M}_{P},
$$

and

$$
\mathcal{M}_{P+X}
\quad\text{vs.}\quad
\mathcal{M}_{P}.
$$

This avoids the situation where only a large combined model is evaluated and the source of improvement cannot be identified.

---

# 19. Primary Evaluation Metric

The primary forecast metric is cross-entropy / log loss:

$$
\mathcal{L}_{log}
=
-
\frac{1}{N}
\sum_{t=1}^{N}
\left[
Y_t\log p_t
+
(1-Y_t)\log(1-p_t)
\right].
$$

Lower values are better.

Log loss is selected because the project aims to generate probabilistic forecasts rather than only class labels.

---

# 20. Secondary Probability Metric

The Brier score is

$$
BS
=
\frac{1}{N}
\sum_{t=1}^{N}
(p_t-Y_t)^2.
$$

Lower values indicate better probability forecasts.

---

# 21. Discrimination Metrics

Secondary discrimination metrics include:

- ROC-AUC,
- precision,
- recall,
- F1,
- balanced accuracy.

These will not replace proper scoring rules as the principal evidence.

---

# 22. Calibration

Forecasts must also be calibrated.

For forecast probability $p$,

$$
\mathbb{P}(Y=1\mid\hat p=p)
\approx p.
$$

Calibration analysis will include:

- reliability curves,
- calibration error,
- probability-bin diagnostics.

A model that predicts direction accurately but produces badly calibrated probabilities is less useful for subsequent trading and prediction-market applications.

---

# 23. Loss Differential

For models $A$ and $B$, define the observation-level loss differential

$$
d_t
=
\ell_{A,t}
-
\ell_{B,t}.
$$

The mean loss difference is

$$
\bar d
=
\frac{1}{N}
\sum_{t=1}^{N}d_t.
$$

For

$$
A=\mathcal{M}_{I+P}
$$

and

$$
B=\mathcal{M}_{P},
$$

evidence for the central hypothesis requires

$$
\bar d<0.
$$

---

# 24. Statistical Uncertainty

Financial prediction errors are serially dependent.

Naive independent-observation standard errors are therefore inappropriate.

Primary uncertainty estimation will use a time-aware resampling method such as:

- moving-block bootstrap,
- stationary bootstrap.

Where appropriate, HAC/Newey-West style estimates may also be reported for loss differentials.

The final method will be fixed before final inference.

---

# 25. Confidence Intervals

For a performance difference

$$
\Delta
=
\mathcal{L}_{A}
-
\mathcal{L}_{B},
$$

the analysis will estimate

$$
CI_{1-\alpha}(\Delta).
$$

A meaningful improvement should not rely solely on a tiny difference in point estimates.

For the primary comparison, stronger evidence exists if

$$
CI_{95\%}(\Delta)<0.
$$

---

# 26. Overlapping Labels

When

$$
h
$$

exceeds the sampling interval, adjacent target observations may overlap.

For example,

$$
R_{t,h}
$$

and

$$
R_{t+1,h}
$$

can contain many of the same future price observations.

This creates dependence in:

- labels,
- forecast errors,
- validation folds.

The validation framework must explicitly account for overlapping horizons.

---

# 27. Chronological Development Structure

The dataset will be divided chronologically into three conceptual regions:

$$
D_{train},
\qquad
D_{development},
\qquad
D_{holdout}.
$$

## Training

Used to estimate model parameters.

## Development

Used for:

- feature decisions,
- model comparison,
- hyperparameter selection,
- horizon selection.

## Locked holdout

Used only after the research specification is frozen.

---

# 28. Locked Holdout Principle

The final holdout is not another validation dataset.

It must not be repeatedly inspected.

Before opening the holdout, the following must be frozen:

- primary target,
- forecast horizons,
- feature definitions,
- data-cleaning rules,
- model families,
- hyperparameter procedure,
- primary metric,
- statistical testing method.

This prevents the holdout from becoming part of the model-development process.

---

# 29. Walk-Forward Evaluation

Within development data, the primary structure is expanding walk-forward evaluation.

For fold $k$,

$$
\mathcal{T}_k
=
\{1,\ldots,t_k\}
$$

and

$$
\mathcal{E}_k
=
\{t_k+1,\ldots,t_k+m\}.
$$

Then

$$
\mathcal{T}_{k}
\subset
\mathcal{T}_{k+1}.
$$

The model always predicts observations occurring after its training period.

---

# 30. Purging and Embargo

Where target intervals overlap training and evaluation boundaries, observations whose label windows cross those boundaries must be removed.

If observation $t$ uses future interval

$$
[t,t+h],
$$

then any training observation whose label window overlaps an evaluation label window may require purging.

An embargo interval may also be inserted around fold boundaries where justified.

This is particularly important at high-frequency and overlapping horizons.

---

# 31. Preprocessing Inside Each Fold

For every fold, all transformations are estimated using the training subset only.

This includes:

$$
\text{scaling},
$$

$$
\text{imputation},
$$

$$
\text{feature selection},
$$

$$
\text{PCA},
$$

$$
\text{embedding reduction},
$$

$$
\text{hyperparameter tuning}.
$$

Formally,

$$
\hat T_k
=
Fit
\left(
\mathcal{T}_k
\right).
$$

The transformation applied to evaluation data is

$$
X_{\mathcal{E}_k}^{*}
=
\hat T_k
\left(
X_{\mathcal{E}_k}
\right).
$$

---

# 32. Hyperparameter Search

Hyperparameters will be selected only inside the development process.

The final holdout cannot influence:

- tree depth,
- learning rate,
- regularization,
- hidden-layer size,
- embedding dimensions,
- ensemble weights,
- probability thresholds.

Search spaces should be defined before tuning.

---

# 33. Multiple Testing Control

The project may evaluate many:

- features,
- domains,
- models,
- horizons.

This creates selection risk.

The research log will therefore record all material experiments rather than only successful results.

Where formal multiple-hypothesis testing becomes extensive, correction procedures or false-discovery controls will be considered.

The primary hypothesis family will remain deliberately small.

---

# 34. Regime Analysis

After establishing a full-sample predictive result, performance may be decomposed by market regime.

Possible states include:

- high / low volatility,
- high / low liquidity,
- trend / mean reversion,
- risk-on / risk-off.

The correct sequence is:

$$
\text{establish signal}
\rightarrow
\text{study regime dependence},
$$

not

$$
\text{search regimes until signal appears}.
$$

---

# 35. Feature Stability

A predictive feature should not be judged solely by global importance.

For feature $j$, evaluate whether its contribution is stable across:

- time,
- folds,
- horizons,
- regimes.

A feature that contributes only during one isolated historical period will be treated cautiously.

---

# 36. Economic Translation

Forecast evaluation and trading evaluation are separate stages.

Only after a predictive model is established will forecasts be translated into positions.

For probability $p_t$,

$$
a_t
=
\begin{cases}
+1, & p_t>\theta_U,\\
0, & \theta_L\leq p_t\leq\theta_U,\\
-1, & p_t<\theta_L.
\end{cases}
$$

The no-trade region reduces trading on low-confidence predictions.

---

# 37. Trading Costs

Economic analysis must include:

- bid-ask spread,
- commissions,
- slippage,
- market impact where relevant,
- funding costs for derivatives where relevant.

Net return is approximately

$$
R_t^{net}
=
R_t^{gross}
-
C_t.
$$

The precise cost function will depend on venue and instrument.

---

# 38. Economic Metrics

Secondary economic metrics may include:

- cumulative return,
- volatility,
- Sharpe ratio,
- maximum drawdown,
- turnover,
- hit rate,
- average return per trade,
- tail loss.

These are secondary to the primary predictive hypothesis.

The project is first a study of information and price formation, not simply a backtest competition.

---

# 39. Prediction-Market Translation

For a binary contract priced at

$$
q_t,
$$

the model's probability estimate is

$$
p_t.
$$

The probability discrepancy is

$$
e_t
=
p_t-q_t.
$$

Later research may test whether positive expected value survives:

- fees,
- spreads,
- liquidity constraints,
- model uncertainty.

This extension will not affect the core Bitcoin forecasting experiment.

---

# 40. Model Acceptance Rule

A complex model is retained only if it improves the research system in a defensible way.

Possible reasons include:

1. lower out-of-sample log loss,
2. improved calibration,
3. stable contribution across folds,
4. incremental information beyond simpler models,
5. economically meaningful improvement.

Complexity alone is not evidence.

---

# 41. Central Success Criteria

The principal research claim is supported if the final locked-holdout analysis shows that

$$
\mathcal{M}_{I+P}
$$

has lower predictive loss than

$$
\mathcal{M}_{P}
$$

with an improvement that is statistically and economically meaningful.

The stronger claim is supported if

$$
\mathcal{M}_{I}
$$

also outperforms

$$
\mathcal{M}_{P}.
$$

---

# 42. Domain-Level Success Criteria

For information domain $j$, evidence of incremental information exists when

$$
\mathcal{L}
\left(
\mathcal{M}_{P+j}
\right)
<
\mathcal{L}
\left(
\mathcal{M}_{P}
\right)
$$

and the improvement is stable enough to survive uncertainty analysis.

This directly measures whether domain $j$ adds information beyond historical price.

---

# 43. Null Result

The research explicitly permits

$$
\mathcal{L}
\left(
\mathcal{M}_{I+P}
\right)
\geq
\mathcal{L}
\left(
\mathcal{M}_{P}
\right).
$$

Such a result would imply that the selected observable information channels do not provide reliable incremental prediction beyond price history under the tested design.

This would remain a valid research result.

---

# 44. Strong Efficient-Assimilation Result

If

$$
\mathcal{M}_{P}
\approx
\mathcal{M}_{I+P}
$$

across several information domains and horizons, the evidence would support the interpretation that observable information is incorporated rapidly into Bitcoin price.

---

# 45. Partial-Adjustment Result

If information variables improve prediction primarily at short horizons before losing usefulness later, this would support a temporary information-assimilation interpretation.

For example,

$$
\Delta(h)>0
$$

for small $h$,

but

$$
\Delta(h)\rightarrow0
$$

as $h$ increases.

This would allow the project to estimate an empirical information-decay structure.

---

# 46. Research Logging

Every material experiment should eventually record:

```text
experiment_id
dataset_version
feature_set
target_definition
horizon
train_period
evaluation_period
model
hyperparameters
random_seed
primary_metric
secondary_metrics
code_version
