# Derivation I — Conditional Information and Predictive Value

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

This document develops the mathematical foundation for the central hypothesis of the research programme.

The objective is not merely to state that non-price information may predict Bitcoin returns.

We want to establish precisely what it means for an information source to contain predictive content beyond historical price and how that concept connects to probabilistic forecasting loss.

The central quantity is

$$
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right),
$$

where:

- $Y_{t,h}$ is the future Bitcoin outcome,
- $\mathbf{p}_t$ is the price-history information state,
- $\mathbf{i}_t$ is the non-price information state.

The derivation below shows that conditional mutual information can be interpreted as the theoretical reduction in optimal logarithmic prediction loss obtained by adding $\mathbf{i}_t$ to $\mathbf{p}_t$.

---

# 1. Prediction Target

Let

$$
R_{t,h}
=
\log
\left(
\frac{S_{t+h}}{S_t}
\right)
$$

denote the Bitcoin forward log return over horizon $h$.

Define the binary target

$$
Y_{t,h}
=
\mathbf{1}
\left\{
R_{t,h}>\tau_h
\right\}.
$$

For compactness, write

$$
Y=Y_{t,h}.
$$

Then

$$
Y\in\{0,1\}.
$$

---

# 2. Competing Information Sets

Let

$$
P=\mathbf{p}_t
$$

denote information derived from historical Bitcoin price.

Let

$$
I=\mathbf{i}_t
$$

denote additional non-price information.

Examples include:

$$
I
=
[
O_t,N_t,C_t,X_t,V_t].
$$

The research compares the conditional distributions

$$
\mathbb{P}(Y\mid P)
$$

and

$$
\mathbb{P}(Y\mid P,I).
$$

The second model has access to everything available to the first model plus additional information.

---

# 3. Conditional Entropy

The entropy of $Y$ is

$$
H(Y)
=
-
\sum_y
p(y)\log p(y).
$$

For a Bernoulli outcome,

$$
H(Y)
=
-
p\log p
-
(1-p)\log(1-p).
$$

Conditional entropy is

$$
H(Y\mid P)
=
-
\mathbb{E}
\left[
\log p(Y\mid P)
\right].
$$

Likewise,

$$
H(Y\mid P,I)
=
-
\mathbb{E}
\left[
\log p(Y\mid P,I)
\right].
$$

Interpretation:

$$
H(Y\mid P)
$$

is the remaining uncertainty about the future outcome after observing price information.

Similarly,

$$
H(Y\mid P,I)
$$

is the remaining uncertainty after observing both price and additional information.

---

# 4. Conditioning Cannot Increase Information Uncertainty

A fundamental information-theoretic result is

$$
H(Y\mid P,I)
\leq
H(Y\mid P).
$$

Therefore,

$$
H(Y\mid P)
-
H(Y\mid P,I)
\geq0.
$$

This difference is precisely conditional mutual information.

---

# 5. Conditional Mutual Information

By definition,

$$
I(Y;I\mid P)
=
H(Y\mid P)
-
H(Y\mid P,I).
$$

Thus

$$
\boxed{
I(Y;I\mid P)
=
H(Y\mid P)
-
H(Y\mid P,I)
}
$$

with

$$
I(Y;I\mid P)\geq0.
$$

This quantity measures how much uncertainty concerning $Y$ is removed by observing $I$ once $P$ is already known.

---

# 6. Interpretation for the Research Question

The central research hypothesis is

$$
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
>0.
$$

This means

$$
H
\left(
Y_{t,h}
\mid
\mathbf{p}_t,\mathbf{i}_t
\right)
<
H
\left(
Y_{t,h}
\mid
\mathbf{p}_t
\right).
$$

Therefore non-price information reduces uncertainty about the future Bitcoin outcome beyond what price history already explains.

This is a more precise claim than saying:

> information is correlated with returns.

Correlation is neither necessary nor sufficient for this hypothesis.

The conditional-information formulation can detect nonlinear dependence and explicitly controls for the information already contained in price history.

---

# 7. Conditional Mutual Information as a Likelihood Ratio

Conditional mutual information may also be written as

$$
I(Y;I\mid P)
=
\mathbb{E}
\left[
\log
\frac{
p(Y\mid P,I)
}{
p(Y\mid P)
}
\right].
$$

Derivation:

Start from

$$
I(Y;I\mid P)
=
\mathbb{E}
\left[
\log
\frac{
p(Y,I\mid P)
}{
p(Y\mid P)p(I\mid P)
}
\right].
$$

Using

$$
p(Y,I\mid P)
=
p(Y\mid I,P)p(I\mid P),
$$

we obtain

$$
I(Y;I\mid P)
=
\mathbb{E}
\left[
\log
\frac{
p(Y\mid I,P)p(I\mid P)
}{
p(Y\mid P)p(I\mid P)
}
\right].
$$

Canceling $p(I\mid P)$ gives

$$
\boxed{
I(Y;I\mid P)
=
\mathbb{E}
\left[
\log
\frac{
p(Y\mid P,I)
}{
p(Y\mid P)
}
\right].
}
$$

This formulation is especially important for probabilistic forecasting.

---

# 8. Connection to Log Loss

For a probabilistic forecast $q(Y\mid X)$, define expected log loss as

$$
\mathcal{L}(q)
=
-
\mathbb{E}
\left[
\log q(Y\mid X)
\right].
$$

The optimal probabilistic predictor under log loss is the true conditional distribution

$$
q^*(Y\mid X)
=
p(Y\mid X).
$$

Therefore the minimum achievable expected log loss using price information is

$$
\mathcal{L}_{P}^{*}
=
-
\mathbb{E}
\left[
\log p(Y\mid P)
\right].
$$

But this is exactly

$$
H(Y\mid P).
$$

Hence

$$
\boxed{
\mathcal{L}_{P}^{*}
=
H(Y\mid P).
}
$$

---

# 9. Optimal Loss with Additional Information

If the predictor observes both $P$ and $I$, the optimal expected log loss becomes

$$
\mathcal{L}_{P,I}^{*}
=
-
\mathbb{E}
\left[
\log p(Y\mid P,I)
\right].
$$

Thus

$$
\boxed{
\mathcal{L}_{P,I}^{*}
=
H(Y\mid P,I).
}
$$

Subtracting gives

$$
\mathcal{L}_{P}^{*}
-
\mathcal{L}_{P,I}^{*}
=
H(Y\mid P)
-
H(Y\mid P,I).
$$

Therefore

$$
\boxed{
\mathcal{L}_{P}^{*}
-
\mathcal{L}_{P,I}^{*}
=
I(Y;I\mid P).
}
$$

This is one of the central mathematical results of the project.

---

# 10. Meaning of the Result

The equation

$$
\boxed{
I(Y;I\mid P)
=
\mathcal{L}_{P}^{*}
-
\mathcal{L}_{P,I}^{*}
}
$$

provides a direct bridge between information theory and empirical forecasting.

If

$$
I(Y;I\mid P)>0,
$$

then an ideal model with access to $I$ can achieve lower expected log loss than an ideal model restricted to $P$.

Therefore our empirical comparison

$$
\mathcal{M}_{P}
\quad\text{vs.}\quad
\mathcal{M}_{P+I}
$$

can be interpreted as a finite-sample approximation to an information-theoretic question.

---

# 11. The Null Hypothesis

Under the null hypothesis,

$$
Y
\perp
I
\mid
P.
$$

Conditional independence implies

$$
p(Y\mid P,I)
=
p(Y\mid P).
$$

Hence

$$
\frac{
p(Y\mid P,I)
}{
p(Y\mid P)
}
=
1.
$$

Therefore

$$
\log 1=0
$$

and

$$
\boxed{
I(Y;I\mid P)=0.
}
$$

This corresponds directly to

$$
\mathcal{L}_{P}^{*}
=
\mathcal{L}_{P,I}^{*}.
$$

In practical language:

> once historical price is known, the additional information contains no further predictive content.

---

# 12. Efficient-Assimilation Interpretation

Suppose public information is incorporated rapidly into Bitcoin price.

Then much of the information contained in $I_t$ may already be encoded in $P_t$.

In that case,

$$
p(Y\mid P,I)
\approx
p(Y\mid P),
$$

giving

$$
I(Y;I\mid P)
\approx0.
$$

Thus an efficient-information-assimilation interpretation emerges naturally as the null model.

The project therefore does not mathematically assume delayed price response.

It tests for it.

---

# 13. Partial-Assimilation Interpretation

Suppose some component of $I_t$ has not yet been completely incorporated into price.

Then

$$
p(Y\mid P,I)
\neq
p(Y\mid P).
$$

Consequently,

$$
I(Y;I\mid P)>0.
$$

The size of the conditional mutual information measures the theoretical predictive value of the residual information.

---

# 14. Domain-Specific Information Contribution

Let

$$
I
=
(I_1,I_2,\ldots,I_J)
$$

represent several information domains.

For example,

$$
I_1=O_t,
$$

$$
I_2=N_t,
$$

$$
I_3=C_t,
$$

$$
I_4=X_t.
$$

The incremental information supplied by domain $j$ conditional on all others is

$$
I
\left(
Y;
I_j
\mid
P,I_{-j}
\right),
$$

where

$$
I_{-j}
$$

denotes all information domains except $j$.

This becomes

$$
I
\left(
Y;
I_j
\mid
P,I_{-j}
\right)
=
H
\left(
Y\mid P,I_{-j}
\right)
-
H
\left(
Y\mid P,I
\right).
$$

This equation provides the theoretical foundation for domain ablation.

---

# 15. Relation to Empirical Ablation

Suppose the complete model has loss

$$
\mathcal{L}_{full}.
$$

Remove domain $j$ and obtain

$$
\mathcal{L}_{-j}.
$$

Empirically define

$$
\Delta_j
=
\mathcal{L}_{-j}
-
\mathcal{L}_{full}.
$$

At the population optimum,

$$
\Delta_j^{*}
=
I
\left(
Y;
I_j
\mid
P,I_{-j}
\right).
$$

Thus ablation loss has an information-theoretic interpretation.

This is stronger than simply reporting feature importance.

---

# 16. Chain Rule for Information

Conditional mutual information obeys the chain rule.

For two information domains $I_1$ and $I_2$,

$$
I(Y;I_1,I_2\mid P)
=
I(Y;I_1\mid P)
+
I(Y;I_2\mid P,I_1).
$$

More generally,

$$
I
\left(
Y;
I_1,\ldots,I_J
\mid
P
\right)
=
\sum_{j=1}^{J}
I
\left(
Y;
I_j
\mid
P,I_1,\ldots,I_{j-1}
\right).
$$

This shows that the total predictive information contained in the multimodal state can be decomposed sequentially.

---

# 17. Redundant Information

Suppose news and order flow contain the same underlying information.

Then it is possible that

$$
I(Y;N\mid P)>0
$$

and

$$
I(Y;O\mid P)>0,
$$

while

$$
I(Y;N\mid P,O)
\approx0.
$$

In that case, news contains predictive information relative to price but little incremental information once order flow is already observed.

This distinguishes:

$$
\text{predictive information}
$$

from

$$
\text{unique predictive information}.
$$

---

# 18. Complementary Information

Conversely, it is possible that

$$
I(Y;N\mid P)
$$

and

$$
I(Y;O\mid P)
$$

are individually small, while their joint state is informative.

That is,

$$
I(Y;N,O\mid P)
$$

may exceed the information suggested by either source individually.

This motivates multimodal interaction models.

---

# 19. Interaction and Synergy

Suppose two sources only become useful when observed jointly.

A simple conceptual measure of interaction is

$$
\mathcal{S}
=
I(Y;N,O\mid P)
-
I(Y;N\mid P)
-
I(Y;O\mid P).
$$

If

$$
\mathcal{S}>0,
$$

the joint representation contains information not captured by the separate marginal contributions under this simple comparison.

This may indicate information synergy.

The interpretation must be handled carefully because rigorous partial-information decomposition is more subtle than this scalar expression.

---

# 20. Information by Forecast Horizon

Conditional information may depend strongly on forecast horizon.

Define

$$
I_h
=
I
\left(
Y_{t,h};
I_t
\mid
P_t
\right).
$$

Then the information structure can be studied as a function

$$
h
\mapsto I_h.
$$

Possible empirical behavior includes

$$
I_h
\downarrow
$$

as $h$ increases.

This would suggest information is incorporated into price over time.

---

# 21. Information Decay

Suppose incremental predictive information decays exponentially:

$$
I_h
=
I_0e^{-\lambda h}.
$$

Then

$$
\log I_h
=
\log I_0
-
\lambda h.
$$

The information half-life satisfies

$$
I_{h_{1/2}}
=
\frac{I_0}{2}.
$$

Therefore

$$
\frac{1}{2}
=
e^{-\lambda h_{1/2}}.
$$

Taking logarithms,

$$
-\log 2
=
-\lambda h_{1/2}.
$$

Hence

$$
\boxed{
h_{1/2}
=
\frac{\log2}{\lambda}.
}
$$

This provides a potential quantitative description of market information-assimilation speed.

---

# 22. Information Arrival Versus Price Response

Suppose an information event arrives at time

$$
\tau.
$$

Define the conditional predictive-information function

$$
G(\delta)
=
I
\left(
Y_{\tau+\delta,h};
I_{\tau}
\mid
P_{\tau+\delta}
\right).
$$

If price progressively incorporates the event, one would expect

$$
G(\delta)
$$

to decline with delay $\delta$.

The empirical shape of

$$
G(\delta)
$$

can therefore characterize the speed with which the information advantage disappears.

---

# 23. Relation to Granger Predictability

Conditional mutual information and Granger-style predictability are related but not identical.

A traditional Granger test asks whether lagged $X$ improves prediction of $Y$ beyond lagged $Y$ under a specified model.

Conceptually,

$$
X
\text{ Granger-causes }
Y
$$

if

$$
\mathbb{P}
\left(
Y_{t+h}
\mid
\mathcal{F}_t^{Y,X}
\right)
\neq
\mathbb{P}
\left(
Y_{t+h}
\mid
\mathcal{F}_t^{Y}
\right).
$$

This resembles

$$
I
\left(
Y_{t+h};
X_t
\mid
P_t
\right)>0.
$$

Conditional mutual information provides a more general dependence formulation because it is not restricted to a linear regression specification.

Neither concept alone establishes structural causality.

---

# 24. Causality Limitation

A finding

$$
I(Y;I\mid P)>0
$$

does not imply

$$
I\rightarrow Y
$$

in a structural causal sense.

Possible explanations include:

- $I$ causally influences price,
- a latent variable influences both $I$ and price,
- $I$ is an earlier manifestation of the same market process,
- measurement latency creates apparent predictive ordering.

Therefore the research language should distinguish:

$$
\text{predictive information}
$$

from

$$
\text{causal information}.
$$

---

# 25. Finite-Sample Estimation

The population quantity

$$
I(Y;I\mid P)
$$

is generally unknown.

Direct high-dimensional mutual-information estimation can be difficult.

The project therefore need not estimate conditional mutual information directly in the first empirical stage.

Instead, we estimate predictive models

$$
\hat p_P(Y\mid P)
$$

and

$$
\hat p_{P,I}(Y\mid P,I)
$$

and compare their out-of-sample log losses.

Define

$$
\widehat{\Delta}_{log}
=
\widehat{\mathcal{L}}_P
-
\widehat{\mathcal{L}}_{P,I}.
$$

Under adequate model approximation and out-of-sample evaluation,

$$
\widehat{\Delta}_{log}
$$

acts as an empirical proxy for the information gain available from $I$.

---

# 26. Model-Misspecification Decomposition

Suppose the estimated model distribution is

$$
q(Y\mid X)
$$

rather than the true distribution

$$
p(Y\mid X).
$$

Expected log loss is

$$
-\mathbb{E}_p
\left[
\log q(Y\mid X)
\right].
$$

Add and subtract

$$
-\mathbb{E}_p
\left[
\log p(Y\mid X)
\right].
$$

Then

$$
\mathcal{L}(q)
=
H(Y\mid X)
+
\mathbb{E}_{X}
\left[
D_{KL}
\left(
p(Y\mid X)
\Vert
q(Y\mid X)
\right)
\right].
$$

Therefore

$$
\boxed{
\mathcal{L}(q)
=
\text{irreducible conditional uncertainty}
+
\text{model approximation error}.
}
$$

This distinction is extremely important.

---

# 27. Derivation of the KL Decomposition

Starting from

$$
\mathcal{L}(q)
=
-
\mathbb{E}_p[\log q(Y\mid X)],
$$

write

$$
-\log q
=
-\log p
+
\log\frac{p}{q}.
$$

Taking expectations,

$$
\mathcal{L}(q)
=
-
\mathbb{E}_p[\log p(Y\mid X)]
+
\mathbb{E}_p
\left[
\log
\frac{
p(Y\mid X)
}{
q(Y\mid X)
}
\right].
$$

The first term is

$$
H(Y\mid X).
$$

The second is

$$
\mathbb{E}_{X}
\left[
D_{KL}
\left(
p(Y\mid X)
\Vert
q(Y\mid X)
\right)
\right].
$$

Hence

$$
\boxed{
\mathcal{L}(q)
=
H(Y\mid X)
+
\mathbb{E}_{X}
D_{KL}(p\Vert q).
}
$$

Since

$$
D_{KL}(p\Vert q)\geq0,
$$

the true conditional probability distribution minimizes expected log loss.

---

# 28. Why a More Informative Model May Perform Worse

The population relation

$$
H(Y\mid P,I)
\leq
H(Y\mid P)
$$

does not guarantee that an estimated information model will outperform an estimated price model.

For estimated models,

$$
\mathcal{L}(q_{P,I})
=
H(Y\mid P,I)
+
E_{P,I},
$$

while

$$
\mathcal{L}(q_P)
=
H(Y\mid P)
+
E_P,
$$

where

$$
E
=
\mathbb{E}D_{KL}(p\Vert q)
$$

is approximation or estimation error.

The information model wins only when

$$
H(Y\mid P)-H(Y\mid P,I)
>
E_{P,I}-E_P.
$$

Since

$$
H(Y\mid P)-H(Y\mid P,I)
=
I(Y;I\mid P),
$$

we obtain

$$
\boxed{
I(Y;I\mid P)
>
E_{P,I}-E_P.
}
$$

This is a crucial finite-sample result.

---

# 29. Interpretation of the Finite-Sample Condition

Additional information can genuinely exist while an empirical model still fails to exploit it.

Reasons include:

- insufficient observations,
- excessive dimensionality,
- poor feature representation,
- model misspecification,
- noisy measurements,
- non-stationarity,
- weak signal-to-noise ratio.

Therefore failure of a complex multimodal model does not logically prove

$$
I(Y;I\mid P)=0.
$$

It establishes only that the tested empirical system failed to extract reliable incremental information under the specified design.

This distinction must be preserved in the final paper.

---

# 30. Complexity Penalty

Adding information dimensions can reduce theoretical uncertainty while increasing estimation error.

Schematically,

$$
\text{Net Predictive Gain}
=
\text{Information Gain}
-
\text{Estimation Cost}.
$$

Thus increasing feature dimension does not guarantee better out-of-sample performance.

This is one reason the project introduces information domains sequentially rather than concatenating every available variable immediately.

---

# 31. Relation to the Model Ladder

The empirical model ladder now has a clear theoretical interpretation.

### Price model

$$
q_P(Y\mid P)
$$

approximates

$$
p(Y\mid P).
$$

### Information model

$$
q_I(Y\mid I)
$$

approximates

$$
p(Y\mid I).
$$

### Combined model

$$
q_{P,I}(Y\mid P,I)
$$

approximates

$$
p(Y\mid P,I).
$$

The central empirical quantity is

$$
\widehat{\Delta}
=
\widehat{\mathcal{L}}_P
-
\widehat{\mathcal{L}}_{P,I}.
$$

Positive values imply that the combined system produced lower out-of-sample log loss.

---

# 32. Statistical Hypothesis

Define

$$
d_t
=
\ell_{P,t}
-
\ell_{P,I,t}.
$$

Then

$$
\Delta
=
\mathbb{E}[d_t].
$$

The empirical hypothesis can be expressed as

$$
H_0:
\Delta\leq0
$$

against

$$
H_1:
\Delta>0.
$$

Because $d_t$ may be serially dependent, inference must use time-aware uncertainty estimates rather than i.i.d. standard errors.

---

# 33. Information Gain in Bits

When logarithms use base 2,

$$
I(Y;I\mid P)
$$

is measured in bits.

When natural logarithms are used, it is measured in nats.

Conversion is

$$
1\text{ nat}
=
\frac{1}{\ln2}
\text{ bits}.
$$

Therefore

$$
I_{bits}
=
\frac{
I_{nats}
}{
\ln2
}.
$$

This potentially provides an intuitive interpretation of incremental forecasting information.

---

# 34. Relative Information Gain

A normalized measure may be defined as

$$
RIG
=
\frac{
H(Y\mid P)
-
H(Y\mid P,I)
}{
H(Y\mid P)
}.
$$

Thus

$$
RIG
=
\frac{
I(Y;I\mid P)
}{
H(Y\mid P)
}.
$$

This measures the fraction of the uncertainty remaining after conditioning on price that is removed by the additional information state.

Empirical use would require careful estimation.

---

# 35. Research Interpretation

The central theoretical question is therefore not simply:

> Can more variables improve a classifier?

It is:

> Does the non-price information state contain conditional information about future Bitcoin price movement that is not already encoded in the price-history state?

Mathematically,

$$
\boxed{
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)>0.
}
$$

Under logarithmic scoring, this quantity has the direct interpretation

$$
\boxed{
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
=
\mathcal{L}_{P}^{*}
-
\mathcal{L}_{P,I}^{*}.
}
$$

This equation forms the mathematical foundation of the entire empirical programme.

---

# 36. Implications for the Research Design

The derivation produces several concrete requirements.

First, the price-only benchmark must be strong because the hypothesis is conditional on price information.

Second, probabilistic log loss is theoretically preferable to directional accuracy for the principal experiment.

Third, domain ablation is not merely a machine-learning diagnostic; it approximates conditional information contribution.

Fourth, different forecast horizons may reveal the temporal decay of information.

Fifth, model complexity must be controlled because theoretical information gain can be overwhelmed by estimation error.

Finally, point-in-time data integrity is essential because future information would artificially increase apparent conditional information.

---

# 37. Final Mathematical Statement

The project can be summarized as an attempt to estimate whether

$$
\boxed{
H
\left(
Y_{t,h}
\mid
\mathbf{p}_t,
\mathbf{i}_t
\right)
<
H
\left(
Y_{t,h}
\mid
\mathbf{p}_t
\right)
}
$$

in a genuinely out-of-sample, point-in-time setting.

Equivalently,

$$
\boxed{
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
>0.
}
$$

And under the optimal logarithmic scoring rule,

$$
\boxed{
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
=
\mathcal{L}_{P}^{*}
-
\mathcal{L}_{P,I}^{*}.
}
$$

This is the mathematical bridge between the project's information-theoretic hypothesis and its empirical machine-learning experiment.
