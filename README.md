# Latent Information Dynamics and Endogenous Price Formation in Bitcoin

### A Multimodal Framework for Pre-Price-Movement Prediction

## Research Motivation

Most financial forecasting systems treat historical price as the principal representation of market state.

This project starts from a different premise:

> Price is an observable consequence of information processing within a market, rather than the information-generating mechanism itself.

By the time information is fully reflected in price, part of its predictive value may already have been incorporated by market participants.

The research therefore investigates whether observable information that precedes price adjustment can be used to construct a predictive representation of the market.

Rather than asking:

> “What does Bitcoin's past price imply about its next price?”

the project asks:

> “What information exists before the next price movement, and how is that information transmitted into price?”

---

## Research Question

Can information observable prior to a Bitcoin price adjustment — including market microstructure, order flow, news, on-chain activity, cross-market disequilibria, macro-financial conditions and latent market states — predict future Bitcoin movements beyond the predictive information contained in Bitcoin's own historical price?

---

## Core Hypothesis

Let the future directional target over horizon $h$ be

```math
Y_{t,h}
=
\mathbf{1}
\left[
\log\left(\frac{S_{t+h}}{S_t}\right) > \tau_h
\right].
```

We compare three information structures.

### Price-only model

```math
\mathcal{M}_{P}
=
P
\left(
Y_{t,h}=1
\mid
\mathcal{P}_t
\right),
```

where $\mathcal{P}_t$ represents information derived primarily from Bitcoin's own historical price process.

### Information model

```math
\mathcal{M}_{I}
=
P
\left(
Y_{t,h}=1
\mid
\mathcal{I}_t
\right),
```

where

```math
\mathcal{I}_t
=
\{
\mathcal{O}_t,
\mathcal{N}_t,
\mathcal{C}_t,
\mathcal{X}_t,
\mathcal{V}_t
\}.
```

The components represent:

- $\mathcal{O}_t$: market microstructure and order-flow state,
- $\mathcal{N}_t$: news and textual information state,
- $\mathcal{C}_t$: blockchain and on-chain state,
- $\mathcal{X}_t$: cross-market and macro-financial state,
- $\mathcal{V}_t$: latent volatility and market-regime state.

### Combined model

```math
\mathcal{M}_{I+P}
=
P
\left(
Y_{t,h}=1
\mid
\mathcal{I}_t,\mathcal{P}_t
\right).
```

The central empirical question is therefore whether

```math
\mathcal{M}_{I}
\succ
\mathcal{M}_{P}
```

out of sample, and whether

```math
\mathcal{M}_{I+P}
```

contains economically meaningful incremental information beyond either representation alone.

---

## Information Formation Architecture

The proposed research architecture is

```math
\text{Information Generation}
\rightarrow
\text{Information Transmission}
\rightarrow
\text{Latent Market State}
\rightarrow
\text{Price Formation}
\rightarrow
\text{Observed Return}.
```

The aim is not to assume that every data source predicts price.

Instead, each information channel is tested for whether it contributes measurable incremental information about the future market state.

---

## Research Layers

### 1. Market Microstructure

The microstructure layer studies the mechanism through which information becomes incorporated into market prices.

Candidate variables include:

- order-book imbalance,
- bid and ask depth,
- queue asymmetry,
- market-order flow,
- signed trade imbalance,
- cancellation intensity,
- spread,
- liquidity,
- price impact,
- order-book resilience,
- cross-exchange price discovery.

---

### 2. Machine-Readable Information

Financial news and public information will be transformed into structured numerical representations.

Candidate information includes:

- event classification,
- sentiment,
- uncertainty,
- novelty,
- entity identification,
- relevance,
- directional implication,
- temporal decay,
- source agreement or disagreement.

LLMs and financial language models are treated primarily as information extraction systems rather than autonomous trading agents.

---

### 3. Blockchain and Crypto-Economic State

Bitcoin provides a public transaction network that may contain information not directly observable in traditional financial markets.

Candidate variables include:

- exchange inflows and outflows,
- miner activity,
- network activity,
- transaction flows,
- hash rate,
- mining difficulty,
- stablecoin liquidity,
- large-holder activity,
- ETF and institutional flows where available.

---

### 4. Cross-Market Information

Bitcoin does not trade in isolation.

The research will test whether disequilibria or information transmission from other markets precede Bitcoin adjustment.

Candidate markets include:

- US interest rates,
- Treasury markets,
- US dollar,
- gold,
- oil and energy,
- equity indices,
- volatility markets,
- Ethereum and the wider crypto complex,
- global liquidity variables.

Methods may include:

- cointegration,
- vector error correction,
- lead-lag analysis,
- conditional dependence,
- regime-dependent relationships,
- information-flow measures.

---

### 5. Latent Information State

The observable information channels may be treated as noisy measurements of a lower-dimensional hidden market state.

A general representation is

```math
Z_t
=
f
\left(
\mathcal{O}_t,
\mathcal{N}_t,
\mathcal{C}_t,
\mathcal{X}_t,
\mathcal{V}_t
\right),
```

where $Z_t$ represents the latent information state.

Candidate approaches include:

- state-space models,
- hidden regimes,
- principal components,
- autoencoders,
- representation learning,
- neural feature fusion.

These methods will only be introduced where they improve the empirical model rather than for methodological complexity alone.

---

## Prediction Layer

The project will evaluate several model families rather than assuming that deep learning must dominate simpler statistical methods.

Initial benchmarks will include:

- logistic regression,
- regularized linear classifiers,
- random forests,
- gradient boosting,
- support vector machines,
- ensemble classifiers.

More complex models may include:

- LSTM / recurrent architectures,
- neural networks,
- multimodal representation networks,
- attention-based architectures,
- stacked or blended specialist models.

A possible architecture is

```math
p_t^{(O)},\;
p_t^{(N)},\;
p_t^{(C)},\;
p_t^{(X)}
\rightarrow
g(\cdot)
\rightarrow
P(Y_{t,h}=1),
```

where specialist models estimate information from separate domains and a meta-model performs information fusion.

---

## Experimental Design

The project will explicitly compare:

1. Price-only models
2. Microstructure-only models
3. News-only models
4. On-chain-only models
5. Cross-market-only models
6. Information-only multimodal models
7. Price + information models

Ablation studies will test the incremental value of each information source.

For example,

```math
\Delta_j
=
L
\left(
\mathcal{M}_{I\setminus j}
\right)
-
L
\left(
\mathcal{M}_{I}
\right)
```

measures the predictive contribution of information block $j$.

---

## Research Integrity

The empirical design will be strictly point-in-time.

Particular attention will be paid to:

- look-ahead bias,
- publication timestamps,
- feature leakage,
- train/test contamination,
- asynchronous data sources,
- revision-sensitive macro data,
- multiple testing,
- hyperparameter overfitting,
- transaction costs,
- probability calibration.

Model development will use chronological validation and eventually walk-forward evaluation.

---

## Intended Contribution

The intended contribution is not another Bitcoin price-prediction model.

The broader research objective is to investigate whether financial price formation can be modelled through the information processes that precede observable price adjustment.

If successful, the study would provide evidence that a multimodal information state contains predictive information that is not captured by the asset's own price history alone.

If the hypothesis fails, that result is equally informative: it would provide evidence about how rapidly Bitcoin markets incorporate publicly observable information.

---

## Current Status

**Stage 1 — Research formulation**

Current work focuses on:

- formalizing the hypotheses,
- developing the mathematical framework,
- mapping the literature,
- defining the information universe,
- specifying the experimental protocol.

Implementation will follow after the research design is sufficiently constrained.

---

## Planned Research Documents

- `docs/RESEARCH_PROPOSAL.md`
- `docs/MATHEMATICAL_FRAMEWORK.md`
- `docs/LITERATURE_MAP.md`
- `docs/DATA_ARCHITECTURE.md`
- `docs/EXPERIMENT_DESIGN.md`
- `docs/VALIDATION_PROTOCOL.md`

---

## Research Direction

The project is designed as an evolving research programme rather than a fixed trading strategy.

The initial Bitcoin study provides a controlled environment in which to investigate a more general question:

> Can the dynamics of information formation and transmission be used to model financial price formation before the resulting movement becomes fully observable in price?
