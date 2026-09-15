# Research Proposal

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

### A Multimodal Framework for Pre-Price-Movement Prediction

## 1. Research Problem

Financial forecasting commonly treats historical price as the primary representation of market state.

This project investigates a different hypothesis: price is the observable outcome of an underlying information-processing mechanism.

Information may originate in:

- order flow and market microstructure,
- public news and narrative information,
- blockchain activity,
- institutional and ETF flows,
- macro-financial conditions,
- related financial markets,
- latent liquidity and volatility states.

If these variables evolve before a corresponding Bitcoin price adjustment, then they may contain predictive information that is not yet fully represented in price.

The project therefore studies the process

```math
\text{Information}
\rightarrow
\text{Market State}
\rightarrow
\text{Price Formation}
```

rather than treating price history itself as the sole forecasting mechanism.

---

## 2. Primary Research Question

Can information observable before a Bitcoin price adjustment predict future Bitcoin directional movements beyond the predictive information contained in Bitcoin's own historical price?

---

## 3. Secondary Research Questions

The project will investigate:

1. Which information channels contain statistically significant predictive information before Bitcoin price adjustment?

2. Does order-book and order-flow information reveal short-horizon price formation earlier than price-derived indicators?

3. Can news and narrative information be transformed into measurable financial state variables using financial language models and large language models?

4. Do blockchain activity and crypto-economic variables contain incremental predictive information?

5. Do disequilibria between Bitcoin and related financial markets lead subsequent Bitcoin adjustment?

6. Can several weak information channels be combined into a stronger latent information representation?

7. Does the predictive contribution of individual information channels change across market regimes?

8. Does an information-based model outperform a price-only benchmark under strict out-of-sample testing?

---

## 4. Main Hypothesis

Define the future Bitcoin return over horizon $h$ as

```math
r_{t,t+h}
=
\log
\left(
\frac{S_{t+h}}{S_t}
\right).
```

The directional target is

```math
Y_{t,h}
=
\mathbf{1}
\left(
r_{t,t+h}>\tau_h
\right),
```

where $\tau_h$ may be zero or an economically meaningful threshold incorporating noise and trading costs.

Let

```math
\mathcal{P}_t
```

denote information derived primarily from Bitcoin's historical price.

Let

```math
\mathcal{I}_t
=
\{
\mathcal{O}_t,
\mathcal{N}_t,
\mathcal{C}_t,
\mathcal{X}_t,
\mathcal{V}_t
\}
```

denote the non-price information state.

The central hypothesis is

```math
H_1:
\quad
P(Y_{t,h}=1\mid\mathcal{I}_t)
```

contains incremental predictive information beyond

```math
P(Y_{t,h}=1\mid\mathcal{P}_t).
```

Operationally, we test whether

```math
\mathcal{M}_I
\succ
\mathcal{M}_P
```

under strictly out-of-sample evaluation.

---

## 5. Information Channels

### 5.1 Market Microstructure

Candidate variables include:

- order-book imbalance,
- bid and ask depth,
- queue imbalance,
- signed order flow,
- aggressive buy and sell volume,
- cancellation intensity,
- bid-ask spread,
- liquidity,
- order-book slope,
- price impact,
- resilience,
- cross-exchange price discovery.

This layer investigates how information becomes incorporated into traded prices.

---

### 5.2 News and Narrative Information

Textual information will be transformed into structured quantitative variables.

Candidate features include:

- sentiment,
- event type,
- entity,
- relevance,
- novelty,
- uncertainty,
- directional implication,
- source credibility,
- disagreement across sources,
- information decay.

Financial language models and LLMs will primarily operate as information extraction systems.

They will not be assumed to generate trading decisions directly.

---

### 5.3 Blockchain and Crypto-Economic Information

Potential features include:

- exchange inflows and outflows,
- miner flows,
- transaction activity,
- active addresses,
- large-holder activity,
- hash rate,
- mining difficulty,
- transaction fees,
- stablecoin liquidity,
- ETF and institutional flows.

These variables provide observable information about the underlying Bitcoin network and capital flows.

---

### 5.4 Cross-Market Information

Candidate explanatory markets include:

- US Treasury yields,
- interest-rate expectations,
- US dollar,
- gold,
- oil and energy,
- equity indices,
- volatility indices,
- Ethereum,
- wider crypto markets,
- global liquidity indicators.

Relationships will not be assumed to be static.

The project will test:

- correlation,
- conditional dependence,
- lead-lag structure,
- cointegration,
- error correction,
- regime dependence.

---

### 5.5 Latent Market State

Observed variables may be noisy measurements of a smaller hidden information state.

A general representation is

```math
Z_t
=
f(\mathcal{I}_t).
```

Candidate approaches include:

- principal components,
- state-space models,
- hidden-state models,
- autoencoders,
- neural representation learning,
- multimodal feature fusion.

The latent-state framework will only be retained if supported empirically.

---

## 6. Model Architecture

The study will begin with interpretable statistical and machine-learning baselines.

Candidate models include:

- logistic regression,
- regularized classifiers,
- random forests,
- gradient boosting,
- support vector machines.

More complex models may subsequently include:

- neural networks,
- LSTM models,
- attention models,
- specialist domain models,
- stacking,
- blending,
- meta-learning.

A possible multimodal structure is

```math
f_O(\mathcal{O}_t)
\rightarrow p_t^O
```

```math
f_N(\mathcal{N}_t)
\rightarrow p_t^N
```

```math
f_C(\mathcal{C}_t)
\rightarrow p_t^C
```

```math
f_X(\mathcal{X}_t)
\rightarrow p_t^X
```

followed by

```math
P(Y_{t,h}=1)
=
g
\left(
p_t^O,
p_t^N,
p_t^C,
p_t^X
\right).
```

This permits each information source to be modelled according to its own structure before information fusion.

---

## 7. Benchmark Design

At minimum, the empirical study will compare:

```math
\mathcal{M}_P
```

Price-only benchmark

against

```math
\mathcal{M}_I
```

Information-only model

and

```math
\mathcal{M}_{I+P}.
```

Combined information and price model.

Additional single-domain benchmarks will isolate:

- microstructure,
- news,
- on-chain,
- cross-market,
- volatility/regime information.

---

## 8. Ablation Analysis

The contribution of each information block will be evaluated through ablation.

For information block $j$,

```math
\Delta_j
=
L(\mathcal{M}_{I\setminus j})
-
L(\mathcal{M}_I),
```

where $L$ is an appropriate predictive loss.

This allows the project to answer not only whether the full model works, but why.

---

## 9. Temporal Integrity

All features must satisfy

```math
X_t \in \mathcal{F}_t,
```

meaning that each variable used for prediction must have been observable at the prediction timestamp.

Particular attention will be given to:

- news publication timestamps,
- delayed market feeds,
- macroeconomic data revisions,
- blockchain confirmation times,
- exchange clock differences,
- asynchronous market observations,
- forward-filled information,
- feature normalization,
- model-selection leakage.

---

## 10. Validation Framework

The project will use chronological validation rather than random train-test splitting.

Later stages will include:

- expanding walk-forward evaluation,
- rolling evaluation,
- purged validation where required,
- probability calibration,
- multiple-horizon testing,
- regime-conditioned testing,
- bootstrap uncertainty,
- economic-value analysis.

Prediction quality will not be assessed using accuracy alone.

Possible metrics include:

- log loss,
- Brier score,
- ROC-AUC,
- precision and recall,
- calibration error,
- information coefficient,
- directional accuracy,
- economic performance after costs.

---

## 11. Scientific Falsifiability

The hypothesis is not that non-price data must predict Bitcoin.

The project explicitly permits the result

```math
\mathcal{M}_I
\not\succ
\mathcal{M}_P.
```

Such a finding would suggest that publicly observable information is incorporated into Bitcoin prices sufficiently rapidly that the selected information channels provide little incremental forecasting power.

That result would still constitute a meaningful empirical finding about Bitcoin price formation.

---

## 12. Intended Contribution

The project aims to contribute in three areas.

### Market microstructure

Investigate the transmission of information into Bitcoin price.

### Multimodal financial prediction

Combine heterogeneous information sources without assuming that one representation is universally optimal.

### Empirical price formation

Test whether information states preceding observed returns provide measurable incremental predictive content.

The ultimate objective is therefore broader than forecasting Bitcoin.

It is to investigate whether financial prediction can be reframed as a problem of modelling the formation and transmission of information before it is completely reflected in observable price.
