# Project Roadmap

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

This document is the master execution plan for the research project.

Its purpose is to prevent the project from becoming a sequence of disconnected experiments or improvised modelling decisions.

The intended research path is:

$$
\text{Theory}
\rightarrow
\text{Mathematical Derivation}
\rightarrow
\text{Data Engineering}
\rightarrow
\text{Information Measurement}
\rightarrow
\text{Predictive Modelling}
\rightarrow
\text{Scientific Validation}
\rightarrow
\text{Economic Translation}
\rightarrow
\text{Research Paper}.
$$

The central research question remains:

> Does information observable before Bitcoin price adjustment contain stable predictive information beyond Bitcoin's own historical price?

The principal comparison is

$$
\mathcal{M}_{P}
\quad\text{vs.}\quad
\mathcal{M}_{I}
\quad\text{vs.}\quad
\mathcal{M}_{I+P}.
$$

---

# Phase I — Research Foundation

## Commit 1 — Research Thesis and Proposal

Status: COMPLETE

Deliverables:

- README
- research motivation
- primary research question
- information-first hypothesis
- price-only benchmark
- multimodal information architecture
- falsifiable framing

---

## Commit 2 — Mathematical Framework

Status: COMPLETE

Deliverables:

- probability space
- filtration
- return target
- information-state notation
- order-book variables
- news representation
- blockchain state
- cross-market state
- cointegration
- latent-state model
- regime model
- ensemble probability model
- calibration
- ablation framework

---

## Commit 3 — Literature Architecture

Status: COMPLETE

Deliverables:

- efficient-markets counter-hypothesis
- costly-information theory
- market microstructure
- order flow
- cryptocurrency market structure
- financial NLP
- cointegration
- machine learning
- validation literature
- literature-to-hypothesis mapping

---

## Commit 4 — Data Architecture

Status: COMPLETE

Deliverables:

- event time
- publication time
- receipt time
- availability time
- asynchronous data
- point-in-time joins
- raw / processed / feature layers
- data lineage
- dataset manifests
- leakage controls

---

## Commit 5 — Experiment Design

Status: COMPLETE

Deliverables:

- model ladder
- benchmark design
- forecast horizons
- domain experiments
- early / late fusion
- ablation experiments
- locked holdout
- economic translation
- success / failure criteria

---

## Commit 6 — Validation Protocol

Status: COMPLETE

Deliverables:

- walk-forward validation
- purging
- embargo
- preprocessing isolation
- block bootstrap
- HAC robustness
- calibration
- multiple-testing discipline
- holdout freeze
- result-admission standard

---

## Commit 7 — Information-Theoretic Derivation

Status: COMPLETE

Deliverables:

- conditional entropy
- conditional mutual information
- likelihood-ratio interpretation
- log-loss derivation
- KL-divergence decomposition
- finite-sample estimation error
- information decay
- information half-life
- domain-level information contribution

Central result:

$$
I(Y;I\mid P)
=
\mathcal{L}_{P}^{*}
-
\mathcal{L}_{P,I}^{*}.
$$

---

# Phase II — Mathematical Core

The mathematical phase should establish the mechanisms underlying the empirical system before complex modelling begins.

---

## Commit 8 — Master Project Roadmap

Status: CURRENT

Purpose:

Create the permanent research execution map.

---

## Commit 9 — Market Microstructure Derivation

Derive the relationship

$$
\text{order arrivals}
\rightarrow
\text{order-flow imbalance}
\rightarrow
\text{liquidity pressure}
\rightarrow
\text{price impact}.
$$

Topics:

- bid and ask processes
- midprice
- microprice
- order-book imbalance
- signed trade flow
- order-flow imbalance
- depth dependence
- linear price-impact approximation
- temporary versus permanent impact
- adverse selection
- price discovery
- cross-venue information transmission

Target derivation:

$$
\Delta m_t
\approx
\beta_t OFI_t
+
\varepsilon_t
$$

with

$$
\beta_t
\propto
\frac{1}{Depth_t}.
$$

---

## Commit 10 — Cointegration and Error-Correction Derivation

Develop from first principles:

$$
X_t\sim I(1),
\qquad
Y_t\sim I(1),
$$

yet

$$
Y_t-\beta X_t\sim I(0).
$$

Derive:

- spurious regression intuition
- cointegrating vector
- equilibrium error
- Engle-Granger representation
- ECM
- multivariate VECM
- adjustment coefficients
- weak exogeneity
- interpretation of cross-market lead-lag

Core structure:

$$
\Delta Y_t
=
\alpha
\left(
Y_{t-1}
-
\beta X_{t-1}
\right)
+
\Gamma\Delta X_t
+
\varepsilon_t.
$$

---

## Commit 11 — State-Space and Latent Information Derivation

Develop:

$$
z_t
=
Az_{t-1}
+
Bu_t
+
\eta_t,
$$

$$
y_t
=
Cz_t
+
\epsilon_t.
$$

Topics:

- hidden state
- observation equation
- transition equation
- state uncertainty
- Kalman filtering
- prediction step
- update step
- innovation
- Kalman gain
- likelihood
- financial interpretation

Possible nonlinear extensions:

- EKF
- particle filters
- nonlinear latent representations

---

## Commit 12 — Regime-Switching Derivation

Develop a hidden-state model

$$
S_t\in\{1,\ldots,K\}.
$$

Derive:

- Markov transition matrix
- filtering probability
- prediction probability
- regime-conditioned likelihood
- regime persistence
- expected regime duration

Example:

$$
P(S_t=j\mid S_{t-1}=i)=p_{ij}.
$$

Research application:

$$
P(Y_{t,h}=1\mid I_t,S_t=k).
$$

---

## Commit 13 — Probabilistic Fusion Derivation

Derive multimodal probability combination.

Topics:

- convex probability averaging
- log-odds combination
- logistic stacking
- Bayesian interpretation
- conditional independence assumptions
- specialist models
- calibration effects

Core form:

$$
\operatorname{logit}(p_t)
=
\beta_0
+
\sum_j
\beta_j
\operatorname{logit}
\left(
p_t^{(j)}
\right).
$$

---

## Commit 14 — Statistical Inference Derivation

Develop the mathematics behind:

- serial dependence
- overlapping horizons
- long-run variance
- HAC covariance
- block bootstrap
- confidence intervals
- forecast-loss differentials

Loss difference:

$$
d_t
=
\ell_{A,t}
-
\ell_{B,t}.
$$

Long-run variance:

$$
\Omega
=
\gamma_0
+
2
\sum_{k=1}^{\infty}
\gamma_k.
$$

---

## Commit 15 — Calibration and Proper Scoring Rules

Develop:

- Bernoulli likelihood
- cross-entropy
- Brier score
- calibration
- discrimination
- reliability
- sharpness
- proper scoring rules

Derive why minimizing expected log loss recovers the true conditional probability.

---

# Phase III — Research Infrastructure

## Commit 16 — Python Package Architecture

Create:

```text
src/
    data/
    features/
    models/
    validation/
    statistics/
    utils/
