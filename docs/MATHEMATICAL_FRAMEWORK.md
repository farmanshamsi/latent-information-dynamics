# Mathematical Framework

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

This document defines the mathematical spine of the research programme.

The central distinction is between the observable price process and the information-generating state that may precede price adjustment.

The project does **not** assume that historical price contains no information. It tests whether non-price information contains incremental predictive content beyond that already encoded in Bitcoin's own price history.

---

# 1. Probability Space and Information Structure

Let

$$
(\Omega,\mathcal{F},\{\mathcal{F}_t\}_{t\geq 0},\mathbb{P})
$$

be a filtered probability space satisfying the usual conditions.

The filtration $\mathcal{F}_t$ represents all information genuinely observable by time $t$.

Any predictor used at time $t$ must be measurable with respect to $\mathcal{F}_t$:

$$
X_t \in \mathcal{F}_t.
$$

More precisely, if $X_t$ is a random variable,

$$
X_t \text{ is } \mathcal{F}_t\text{-measurable}.
$$

This is the mathematical statement of the point-in-time constraint.

---

# 2. Bitcoin Price Process

Let

$$
S_t > 0
$$

denote the Bitcoin spot price at time $t$.

For forecast horizon $h>0$, define the forward log return

$$
R_{t,h}
=
\log\left(
\frac{S_{t+h}}{S_t}
\right).
$$

The basic directional target is

$$
Y_{t,h}
=
\mathbf{1}
\left\{
R_{t,h}>\tau_h
\right\},
$$

where $\tau_h$ is a horizon-dependent decision threshold.

The simplest case is

$$
\tau_h=0.
$$

A cost-aware formulation may instead impose

$$
\tau_h>0
$$

to exclude movements too small to be economically meaningful.

A three-state extension is

$$
Y_{t,h}
=
\begin{cases}
-1, & R_{t,h}<-\tau_h,\\[4pt]
0, & |R_{t,h}|\leq\tau_h,\\[4pt]
1, & R_{t,h}>\tau_h.
\end{cases}
$$

---

# 3. Price and Information Representations

Let

$$
\mathbf{p}_t
$$

denote the vector of features derived primarily from Bitcoin's own historical price process.

Define the non-price information vector as

$$
\mathbf{i}_t
=
\begin{bmatrix}
\mathbf{o}_t\\
\mathbf{n}_t\\
\mathbf{c}_t\\
\mathbf{x}_t\\
\mathbf{v}_t
\end{bmatrix},
$$

where:

- $\mathbf{o}_t$ = market microstructure and order-flow state,
- $\mathbf{n}_t$ = news and textual-information state,
- $\mathbf{c}_t$ = blockchain and crypto-economic state,
- $\mathbf{x}_t$ = cross-market and macro-financial state,
- $\mathbf{v}_t$ = volatility, liquidity and regime state.

The principal information-theoretic hypothesis is

$$
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
>0,
$$

where $I(\cdot;\cdot\mid\cdot)$ denotes conditional mutual information.

This asks whether non-price information contains predictive content about the future outcome after conditioning on information already contained in price history.

---

# 4. Competing Predictive Systems

## 4.1 Price-only model

$$
p^{P}_{t,h}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{p}_t
\right).
$$

## 4.2 Information-only model

$$
p^{I}_{t,h}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t
\right).
$$

## 4.3 Combined model

$$
p^{I+P}_{t,h}
=
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,\mathbf{p}_t
\right).
$$

The strong research hypothesis is

$$
\mathcal{M}_{I}
\succ
\mathcal{M}_{P}
$$

out of sample.

A weaker but still important finding would be

$$
\mathcal{M}_{I+P}
\succ
\mathcal{M}_{P}.
$$

That would establish incremental predictive information even if price history remains useful.

---

# 5. Limit-Order-Book State

At time $t$, let the first $K$ bid levels be

$$
\left\{
(b_t^{(k)},q_{t,b}^{(k)})
\right\}_{k=1}^{K},
$$

and the first $K$ ask levels be

$$
\left\{
(a_t^{(k)},q_{t,a}^{(k)})
\right\}_{k=1}^{K}.
$$

The best bid and ask are

$$
b_t=b_t^{(1)},
\qquad
a_t=a_t^{(1)}.
$$

The midquote is

$$
m_t
=
\frac{a_t+b_t}{2},
$$

and the quoted spread is

$$
s_t
=
a_t-b_t.
$$

---

# 6. Order-Book Imbalance

Aggregate bid depth is

$$
D_t^{B}
=
\sum_{k=1}^{K}
q_{t,b}^{(k)},
$$

while aggregate ask depth is

$$
D_t^{A}
=
\sum_{k=1}^{K}
q_{t,a}^{(k)}.
$$

Define normalized order-book imbalance as

$$
OBI_t
=
\frac{
D_t^{B}-D_t^{A}
}{
D_t^{B}+D_t^{A}
}.
$$

Hence

$$
OBI_t\in[-1,1].
$$

Positive values indicate relatively greater displayed bid-side depth.

---

# 7. Signed Trade Flow

Let trade $j$ occurring during interval $t$ have volume $v_j$ and direction

$$
\epsilon_j\in\{-1,+1\},
$$

where

$$
\epsilon_j
=
\begin{cases}
+1, & \text{buyer initiated},\\
-1, & \text{seller initiated}.
\end{cases}
$$

Signed volume is

$$
Q_t
=
\sum_{j\in\mathcal{T}_t}
\epsilon_jv_j.
$$

A normalized trade imbalance is

$$
TI_t
=
\frac{
\sum_{j\in\mathcal{T}_t}\epsilon_jv_j
}{
\sum_{j\in\mathcal{T}_t}v_j
}.
$$

This attempts to measure aggressive trading pressure rather than simply contemporaneous price movement.

---

# 8. Order-Flow Imbalance

A generalized order-flow imbalance can be represented as

$$
OFI_t
=
\Delta Q_t^{B}
-
\Delta Q_t^{A}.
$$

Changes in quoted depth may result from:

- limit-order submissions,
- cancellations,
- executions,
- queue depletion.

The microstructure state may therefore be represented as

$$
\mathbf{o}_t
=
f_O
\left(
OBI_t,
TI_t,
OFI_t,
s_t,
D_t,
C_t,
R_t^{LOB}
\right),
$$

where $C_t$ denotes cancellation intensity and $R_t^{LOB}$ represents order-book resilience.

---

# 9. News Information Mapping

Suppose news item $j$ becomes observable at publication timestamp

$$
\tau_j.
$$

Its text is denoted

$$
x_j.
$$

A language model generates a numerical representation

$$
\mathbf{e}_j
=
\phi(x_j),
\qquad
\mathbf{e}_j\in\mathbb{R}^{d}.
$$

Additional structured quantities may include

$$
s_j=\text{sentiment},
$$

$$
u_j=\text{uncertainty},
$$

$$
\nu_j=\text{novelty},
$$

$$
r_j=\text{relevance},
$$

and

$$
c_j=\text{event category}.
$$

The language model therefore acts as an information-extraction operator

$$
x_j
\longrightarrow
\phi(x_j)
\longrightarrow
\text{structured financial state}.
$$

---

# 10. Temporal Information Decay

News information need not remain equally relevant indefinitely.

For event $j$, define

$$
w_j(t)
=
\exp
\left[
-\lambda_c(t-\tau_j)
\right]
\mathbf{1}_{\{\tau_j\leq t\}},
$$

where $\lambda_c>0$ may depend on event class $c$.

The corresponding information half-life is

$$
h_{1/2}^{(c)}
=
\frac{\ln 2}{\lambda_c}.
$$

The aggregate news state may then be written

$$
\mathbf{n}_t
=
\sum_{\tau_j\leq t}
w_j(t)
\,
\psi
\left(
\mathbf{e}_j,
s_j,
u_j,
\nu_j,
r_j,
c_j
\right).
$$

This allows the persistence of information to be learned rather than assumed.

---

# 11. Information Novelty

Let $\mathbf{e}_j$ denote the embedding of a new event.

Let

$$
\bar{\mathbf{e}}_{j^-}
$$

represent an aggregate embedding of information observable immediately before the event.

Define cosine similarity as

$$
\operatorname{sim}
\left(
\mathbf{e}_j,
\bar{\mathbf{e}}_{j^-}
\right)
=
\frac{
\mathbf{e}_j^\top
\bar{\mathbf{e}}_{j^-}
}{
\|\mathbf{e}_j\|_2
\,
\|\bar{\mathbf{e}}_{j^-}\|_2
}.
$$

A simple novelty statistic is

$$
Novelty_j
=
1-
\operatorname{sim}
\left(
\mathbf{e}_j,
\bar{\mathbf{e}}_{j^-}
\right).
$$

This distinguishes repeated information from genuinely new information.

---

# 12. Blockchain State

Define the crypto-economic state vector as

$$
\mathbf{c}_t
=
\begin{bmatrix}
F_t^{exchange}\\
F_t^{miner}\\
A_t^{network}\\
H_t\\
D_t\\
L_t^{stablecoin}\\
W_t^{large}\\
F_t^{ETF}
\end{bmatrix}.
$$

Possible components include:

- exchange inflows and outflows,
- miner activity,
- network activity,
- hash rate,
- mining difficulty,
- stablecoin liquidity,
- large-holder flows,
- ETF or institutional flows.

These variables are initially treated as observable state variables, not assumed alpha signals.

---

# 13. Cross-Market State

Let

$$
\mathbf{x}_t
=
\begin{bmatrix}
x_t^{rates}\\
x_t^{USD}\\
x_t^{gold}\\
x_t^{oil}\\
x_t^{equity}\\
x_t^{volatility}\\
x_t^{ETH}\\
\vdots
\end{bmatrix}.
$$

The project will investigate:

$$
\text{correlation},
\quad
\text{lead-lag},
\quad
\text{conditional dependence},
\quad
\text{cointegration},
\quad
\text{error correction}.
$$

No static relationship is assumed in advance.

---

# 14. Cointegration

Suppose

$$
y_t\sim I(1)
$$

and

$$
x_t\sim I(1).
$$

If there exists $\beta$ such that

$$
z_t
=
y_t-\alpha-\beta x_t
$$

satisfies

$$
z_t\sim I(0),
$$

then $x_t$ and $y_t$ are cointegrated.

For Bitcoin,

$$
z_t
$$

can be interpreted as a temporary deviation from an estimated long-run cross-market relation.

The relevant question is whether

$$
z_t
$$

predicts subsequent Bitcoin adjustment.

---

# 15. Error-Correction Dynamics

A two-variable error-correction model can be written

$$
\Delta y_t
=
\gamma z_{t-1}
+
\sum_{i=1}^{p}
\phi_i\Delta y_{t-i}
+
\sum_{j=1}^{q}
\theta_j\Delta x_{t-j}
+
\varepsilon_t.
$$

The coefficient

$$
\gamma
$$

measures the rate at which $y_t$ responds to disequilibrium.

If Bitcoin systematically adjusts after related markets move, then

$$
z_{t-1}
$$

may contain information about future Bitcoin returns.

That relationship must be demonstrated empirically.

---

# 16. Multivariate Cointegration

For

$$
\mathbf{Z}_t\in\mathbb{R}^{n},
$$

the vector error-correction representation is

$$
\Delta\mathbf{Z}_t
=
\Pi\mathbf{Z}_{t-1}
+
\sum_{i=1}^{p-1}
\Gamma_i
\Delta\mathbf{Z}_{t-i}
+
\boldsymbol{\varepsilon}_t.
$$

If

$$
\operatorname{rank}(\Pi)=r<n,
$$

then

$$
\Pi
=
\alpha\beta^\top.
$$

Here:

$$
\beta
$$

contains the cointegrating relations, while

$$
\alpha
$$

contains adjustment coefficients.

This permits joint systems involving Bitcoin, Ethereum, rates, dollar, gold, equities, commodities and liquidity variables.

---

# 17. Latent Information State

The observed feature blocks may be noisy measurements of a lower-dimensional hidden state.

Let

$$
\mathbf{z}_t\in\mathbb{R}^{d_z}
$$

denote the latent information state.

A linear state-space representation is

$$
\mathbf{z}_t
=
A\mathbf{z}_{t-1}
+
B\mathbf{u}_t
+
\boldsymbol{\eta}_t,
$$

where

$$
\mathbf{u}_t
=
\begin{bmatrix}
\mathbf{o}_t\\
\mathbf{n}_t\\
\mathbf{c}_t\\
\mathbf{x}_t\\
\mathbf{v}_t
\end{bmatrix},
$$

and

$$
\boldsymbol{\eta}_t
\sim
\mathcal{N}(0,Q).
$$

A return observation equation may be

$$
R_{t,h}
=
\mathbf{c}^{\top}\mathbf{z}_t
+
\varepsilon_{t+h}.
$$

For directional classification,

$$
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{z}_t
\right)
=
\sigma
\left(
\beta_0+
\boldsymbol{\beta}^{\top}\mathbf{z}_t
\right),
$$

where

$$
\sigma(x)
=
\frac{1}{1+e^{-x}}.
$$

---

# 18. Regime-Switching Information Transmission

Let

$$
S_t\in\{1,\ldots,K\}
$$

represent a latent market regime.

The transition matrix satisfies

$$
p_{ij}
=
\mathbb{P}
\left(
S_t=j
\mid
S_{t-1}=i
\right),
$$

with

$$
\sum_{j=1}^{K}p_{ij}=1.
$$

The predictive relation becomes

$$
\mathbb{P}
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,S_t=k
\right)
=
f_k(\mathbf{i}_t).
$$

Possible latent states could correspond to differences in:

- liquidity,
- volatility,
- risk appetite,
- market stress,
- trend persistence.

---

# 19. Specialist Information Encoders

Because each information source has a different statistical structure, define

$$
\mathbf{z}_t^{O}
=
f_O(\mathbf{o}_t),
$$

$$
\mathbf{z}_t^{N}
=
f_N(\mathbf{n}_t),
$$

$$
\mathbf{z}_t^{C}
=
f_C(\mathbf{c}_t),
$$

and

$$
\mathbf{z}_t^{X}
=
f_X(\mathbf{x}_t).
$$

The multimodal latent representation is then

$$
\mathbf{z}_t
=
g
\left(
\mathbf{z}_t^{O},
\mathbf{z}_t^{N},
\mathbf{z}_t^{C},
\mathbf{z}_t^{X}
\right).
$$

---

# 20. Specialist Probability Models

Alternatively, each information source can produce its own probability:

$$
p_t^{O}
=
\mathbb{P}
(Y_{t,h}=1\mid\mathbf{o}_t),
$$

$$
p_t^{N}
=
\mathbb{P}
(Y_{t,h}=1\mid\mathbf{n}_t),
$$

$$
p_t^{C}
=
\mathbb{P}
(Y_{t,h}=1\mid\mathbf{c}_t),
$$

$$
p_t^{X}
=
\mathbb{P}
(Y_{t,h}=1\mid\mathbf{x}_t).
$$

A weighted ensemble is

$$
p_t
=
\sum_{j=1}^{J}
w_jp_t^{(j)},
$$

subject to

$$
w_j\geq0
$$

and

$$
\sum_{j=1}^{J}w_j=1.
$$

---

# 21. Stacked Meta-Model

A more flexible probability-fusion model is

$$
p_t
=
\sigma
\left[
\beta_0
+
\sum_{j=1}^{J}
\beta_j
\operatorname{logit}
\left(
p_t^{(j)}
\right)
\right],
$$

where

$$
\operatorname{logit}(p)
=
\log
\left(
\frac{p}{1-p}
\right).
$$

This provides a probabilistic formulation of stacking.

---

# 22. Classification Loss

For observations $t=1,\ldots,N$, binary cross-entropy is

$$
\mathcal{L}_{CE}
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

This penalizes incorrect and overconfident probability forecasts more appropriately than raw directional accuracy.

---

# 23. Brier Score

Probability accuracy may also be measured using

$$
BS
=
\frac{1}{N}
\sum_{t=1}^{N}
(p_t-Y_t)^2.
$$

A calibrated model should approximately satisfy

$$
\mathbb{P}
(Y_t=1\mid p_t=p)
=
p.
$$

Thus predictions near $0.70$ should correspond to approximately $70\%$ empirical event frequency.

---

# 24. Ablation Analysis

Let

$$
\mathcal{L}_{full}
$$

be the out-of-sample loss of the complete information system.

Let

$$
\mathcal{L}_{-j}
$$

be the corresponding loss after removing information block $j$.

Define

$$
\Delta_j
=
\mathcal{L}_{-j}
-
\mathcal{L}_{full}.
$$

If

$$
\Delta_j>0,
$$

then information block $j$ improves predictive performance.

---

# 25. Incremental Information

A stronger information-theoretic measure is

$$
I
\left(
Y_{t,h};
\mathbf{x}_{j,t}
\mid
\mathbf{i}_{-j,t},
\mathbf{p}_t
\right).
$$

This asks whether information block $j$ contributes knowledge about the future target that is not already contained in the other information sources and historical price.

---

# 26. Point-in-Time Availability

For source $X$, define an availability operator

$$
\mathcal{A}_t(X)
$$

as the most recent observation genuinely available to a market participant by time $t$.

The usable feature is therefore

$$
X_t^{usable}
=
\mathcal{A}_t(X).
$$

This is not necessarily equal to the value stored at timestamp $t$ in a final historical database.

---

# 27. Chronological Out-of-Sample Evaluation

Let

$$
\mathcal{D}_1,\mathcal{D}_2,\ldots,\mathcal{D}_K
$$

be chronologically ordered evaluation blocks.

For fold $k$, define the training set

$$
\mathcal{T}_k
=
\bigcup_{j<k}\mathcal{D}_j.
$$

The evaluation set is

$$
\mathcal{E}_k
=
\mathcal{D}_k.
$$

Thus

$$
\max(\mathcal{T}_k)
<
\min(\mathcal{E}_k).
$$

All preprocessing must be fitted only on $\mathcal{T}_k$.

This includes:

$$
\text{scaling},
\quad
\text{feature selection},
\quad
\text{hyperparameters},
\quad
\text{representation learning}.
$$

---

# 28. Economic Decision Layer

Prediction is separated from trading.

Given forecast probability

$$
p_t
=
\mathbb{P}
(Y_{t,h}=1\mid\mathcal{F}_t),
$$

define

$$
a_t
=
\begin{cases}
+1, & p_t>\theta_U,\\[4pt]
0, & \theta_L\leq p_t\leq\theta_U,\\[4pt]
-1, & p_t<\theta_L.
\end{cases}
$$

The interval

$$
[\theta_L,\theta_U]
$$

creates a no-trade region.

Thresholds must be determined exclusively from training or validation data.

---

# 29. Prediction-Market Extension

Consider a binary contract paying

$$
Y\in\{0,1\}.
$$

Under the research model, its probability-based fair value is

$$
V_t
=
\mathbb{P}
(Y=1\mid\mathcal{F}_t).
$$

If the prediction-market price is

$$
q_t,
$$

define model edge

$$
E_t
=
V_t-q_t.
$$

Trading can therefore be based on discrepancies between estimated probability and market-implied probability.

---

# 30. Core Falsifiable Hypotheses

### Hypothesis 1 — Incremental information

$$
H_1:
\qquad
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
>0.
$$

### Hypothesis 2 — Information versus price

$$
H_2:
\qquad
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

### Hypothesis 3 — Incremental value over price

$$
H_3:
\qquad
\mathcal{L}
\left(
\mathcal{M}_{I+P}
\right)
<
\mathcal{L}
\left(
\mathcal{M}_{P}
\right).
$$

### Hypothesis 4 — Domain contribution

For at least one information domain $j$,

$$
H_4:
\qquad
\Delta_j>0.
$$

### Hypothesis 5 — Regime dependence

For some regimes $k\neq l$,

$$
H_5:
\qquad
\mathbb{P}
(Y_{t,h}=1\mid\mathbf{i}_t,S_t=k)
\neq
\mathbb{P}
(Y_{t,h}=1\mid\mathbf{i}_t,S_t=l).
$$

---

# 31. Research Principle

The project does not equate complexity with information.

A mathematical or machine-learning component will remain in the final architecture only if its incremental contribution survives:

$$
\text{temporal validation}
$$

$$
+
\text{ablation}
$$

$$
+
\text{statistical uncertainty}
$$

$$
+
\text{probability calibration}
$$

$$
+
\text{economic-cost analysis}.
$$

The objective is not to construct the most complicated forecasting model.

The objective is to identify the most defensible representation of information that precedes price formation.
