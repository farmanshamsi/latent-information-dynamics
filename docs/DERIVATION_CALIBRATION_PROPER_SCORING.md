# Derivation X — Calibration, Proper Scoring Rules and Decision-Useful Probabilities

## From Directional Predictions to Economically Meaningful Probability Forecasts

The project does not ultimately need a classifier that merely predicts:

```math
\text{up}
```

or

```math
\text{down}.
```

It requires probabilities of the form

```math
\boxed{
p_{t,h}
=
P
\left(
Y_{t,h}=1
\mid
\mathcal{F}_t
\right).
}
```

The difference is fundamental.

A forecast of

```math
p_{t,h}=0.51
```

and a forecast of

```math
p_{t,h}=0.95
```

produce the same binary class under a threshold of $0.5$.

But they imply radically different confidence, risk and economic action.

The mathematical chain is therefore

```math
\boxed{
\text{Probability Forecast}
\rightarrow
\text{Proper Scoring}
\rightarrow
\text{Calibration}
\rightarrow
\text{Uncertainty}
\rightarrow
\text{Decision Rule}
\rightarrow
\text{Economic Value}.
}
```

---

# 1. Binary Event Probability

Let

```math
Y
\in
\{0,1\}.
```

The true conditional probability is

```math
\boxed{
q
=
P(Y=1\mid X).
}
```

A model reports probability

```math
p
\in
(0,1).
```

The scoring rule should reward the model when

```math
p
```

is close to the true probability

```math
q.
```

---

# 2. Why Accuracy Is Insufficient

Suppose two models produce:

```math
p_A=0.51
```

and

```math
p_B=0.90.
```

If the outcome is

```math
Y=1,
```

both models are classified as correct.

Accuracy therefore assigns both forecasts the same reward.

But model $B$ made a substantially stronger statement.

Probability scoring rules distinguish these forecasts.

---

# 3. Bernoulli Likelihood

For binary outcome $Y$ and forecast probability $p$, the Bernoulli likelihood is

```math
L(p;Y)
=
p^Y
(1-p)^{1-Y}.
```

If

```math
Y=1,
```

then

```math
L=p.
```

If

```math
Y=0,
```

then

```math
L=1-p.
```

---

# 4. Log Likelihood

Taking logarithms,

```math
\log L
=
Y\log p
+
(1-Y)\log(1-p).
```

Negative log likelihood gives log loss:

```math
\boxed{
\ell_{\log}(p,Y)
=
-
\left[
Y\log p
+
(1-Y)\log(1-p)
\right].
}
```

---

# 5. Sample Log Loss

Across $T$ forecasts,

```math
\boxed{
\mathcal{L}_{\log}
=
-
\frac{1}{T}
\sum_{t=1}^{T}
\left[
Y_t\log p_t
+
(1-Y_t)\log(1-p_t)
\right].
}
```

Lower values are better.

This is the principal forecast-quality metric for the project.

---

# 6. Expected Log Loss

Assume the true event probability is

```math
q.
```

The model reports probability $p$.

Expected log loss is

```math
E[
\ell_{\log}(p,Y)
]
=
-q\log p
-
(1-q)\log(1-p).
```

Define

```math
\boxed{
R_{\log}(p;q)
=
-q\log p
-
(1-q)\log(1-p).
}
```

---

# 7. Optimal Probability Under Log Loss

Differentiate with respect to $p$:

```math
\frac{
\partial R_{\log}
}{
\partial p
}
=
-\frac{q}{p}
+
\frac{1-q}{1-p}.
```

Set equal to zero:

```math
-\frac{q}{p}
+
\frac{1-q}{1-p}
=
0.
```

Therefore

```math
\frac{1-q}{1-p}
=
\frac{q}{p}.
```

Cross-multiplying,

```math
p(1-q)
=
q(1-p).
```

Hence

```math
p-pq
=
q-pq.
```

Therefore

```math
\boxed{
p=q.
}
```

---

# 8. Second-Order Condition

Differentiate again:

```math
\frac{
\partial^2 R_{\log}
}{
\partial p^2
}
=
\frac{q}{p^2}
+
\frac{1-q}{(1-p)^2}.
```

For

```math
0<p<1,
```

this quantity is strictly positive.

Therefore

```math
\boxed{
p=q
}
```

is the unique minimum.

This proves that logarithmic loss is **strictly proper**.

---

# 9. Meaning of Strict Propriety

A scoring rule is strictly proper when the forecaster minimizes expected loss by reporting its true probability belief.

Thus the scoring system does not reward strategic distortion.

For log loss:

```math
\boxed{
p^*
=
P(Y=1\mid X).
}
```

This is one reason log loss is appropriate for probabilistic financial forecasting.

---

# 10. Cross-Entropy Interpretation

Expected log loss can be written as cross entropy:

```math
H(q,p)
=
-q\log p
-
(1-q)\log(1-p).
```

The entropy of the true Bernoulli distribution is

```math
H(q)
=
-q\log q
-
(1-q)\log(1-q).
```

The relationship is

```math
\boxed{
H(q,p)
=
H(q)
+
D_{KL}
\left(
Bern(q)
\Vert
Bern(p)
\right).
}
```

---

# 11. Deriving the KL Decomposition

The Bernoulli KL divergence is

```math
D_{KL}
\left(
q\Vert p
\right)
=
q
\log
\frac{q}{p}
+
(1-q)
\log
\frac{1-q}{1-p}.
```

Expand:

```math
D_{KL}(q\Vert p)
=
q\log q
-
q\log p
+
(1-q)\log(1-q)
-
(1-q)\log(1-p).
```

Rearranging,

```math
-q\log p
-
(1-q)\log(1-p)
```

equals

```math
-q\log q
-
(1-q)\log(1-q)
+
D_{KL}(q\Vert p).
```

Therefore

```math
\boxed{
R_{\log}(p;q)
=
H(q)
+
D_{KL}(q\Vert p).
}
```

Because

```math
D_{KL}\geq0,
```

the minimum occurs at

```math
p=q.
```

---

# 12. Connection to the Research Thesis

For feature set $X$,

```math
\mathcal{L}_{\log}^*(X)
=
H(Y\mid X).
```

Thus if

```math
P
```

denotes price information and

```math
I
```

denotes non-price information,

```math
\boxed{
\mathcal{L}_{P}^{*}
-
\mathcal{L}_{P,I}^{*}
=
I(Y;I\mid P).
}
```

This is why log loss is not merely a convenient metric.

It is mathematically linked to the central conditional-information hypothesis.

---

# 13. Extreme Probability Penalty

If the model predicts

```math
p\rightarrow1
```

but

```math
Y=0,
```

then

```math
-\log(1-p)
\rightarrow\infty.
```

Likewise, if

```math
p\rightarrow0
```

while

```math
Y=1,
```

then

```math
-\log p
\rightarrow\infty.
```

Thus log loss strongly penalizes confident mistakes.

---

# 14. Probability Clipping

Numerically, exact probabilities zero and one create infinite loss.

Therefore evaluation may use fixed clipping:

```math
\boxed{
\tilde p
=
\min
\left(
1-\epsilon,
\max(\epsilon,p)
\right).
}
```

For example,

```math
\epsilon=10^{-6}.
```

The clipping rule must be identical across models and fixed before evaluation.

---

# 15. Brier Score

Another proper probability score is the Brier score:

```math
\boxed{
BS
=
(p-Y)^2.
}
```

Across $T$ observations,

```math
\boxed{
BS_T
=
\frac{1}{T}
\sum_{t=1}^{T}
(p_t-Y_t)^2.
}
```

Lower values are better.

---

# 16. Expected Brier Score

Assume again

```math
P(Y=1)=q.
```

Expected Brier loss is

```math
E[(p-Y)^2]
=
q(p-1)^2
+
(1-q)p^2.
```

Expand:

```math
q(p^2-2p+1)
+
(1-q)p^2.
```

Therefore

```math
=
qp^2
-
2qp
+
q
+
p^2
-
qp^2.
```

Hence

```math
\boxed{
E[(p-Y)^2]
=
p^2
-
2qp
+
q.
}
```

---

# 17. Properness of Brier Score

Differentiate:

```math
\frac{\partial}{\partial p}
E[(p-Y)^2]
=
2p-2q.
```

Setting this to zero gives

```math
\boxed{
p=q.
}
```

The second derivative is

```math
2>0.
```

Therefore Brier score is also strictly proper.

---

# 18. Brier Risk Decomposition Around the Truth

Write

```math
p^2-2pq+q
```

as

```math
(p-q)^2
+
q-q^2.
```

Since

```math
q-q^2
=
q(1-q),
```

we obtain

```math
\boxed{
E[(p-Y)^2]
=
(p-q)^2
+
q(1-q).
}
```

The first term is forecast error relative to the true probability.

The second is irreducible Bernoulli uncertainty.

---

# 19. Calibration Definition

A probabilistic forecast is calibrated if

```math
\boxed{
P(Y=1\mid p=q)
=
q.
}
```

More generally,

```math
\boxed{
E[Y\mid p]
=
p.
}
```

Thus among predictions near $0.70$, approximately 70% of outcomes should equal one.

---

# 20. Calibration Versus Discrimination

Calibration asks:

> Are forecast probabilities numerically trustworthy?

Discrimination asks:

> Can the model rank positive outcomes above negative outcomes?

A model may discriminate well but be poorly calibrated.

For example, multiplying every model logit by a large constant preserves ranking while making probabilities excessively extreme.

---

# 21. Calibration Function

Define

```math
m(p)
=
E[Y\mid \hat p=p].
```

Perfect calibration requires

```math
\boxed{
m(p)=p.
}
```

A calibration curve compares estimated

```math
m(p)
```

against the 45-degree line.

---

# 22. Reliability Diagram

Partition probabilities into bins

```math
B_1,\ldots,B_K.
```

For bin $k$, calculate average prediction

```math
\bar p_k
=
\frac{1}{n_k}
\sum_{t\in B_k}
p_t.
```

Calculate empirical event frequency

```math
\bar y_k
=
\frac{1}{n_k}
\sum_{t\in B_k}
Y_t.
```

Calibration requires approximately

```math
\boxed{
\bar p_k
\approx
\bar y_k.
}
```

---

# 23. Expected Calibration Error

One common summary is

```math
\boxed{
ECE
=
\sum_{k=1}^{K}
\frac{n_k}{T}
\left|
\bar p_k-\bar y_k
\right|.
}
```

This is intuitive but has limitations.

It depends on:

- bin definitions,
- number of bins,
- sample size.

Therefore ECE should not be the only calibration measure.

---

# 24. Maximum Calibration Error

Another diagnostic is

```math
\boxed{
MCE
=
\max_k
\left|
\bar p_k-\bar y_k
\right|.
}
```

This identifies the worst observed calibration bin.

It can be noisy when bins contain few observations.

---

# 25. Calibration Intercept and Slope

Suppose model forecast logit is

```math
\ell_t
=
\operatorname{logit}(p_t).
```

Fit calibration model

```math
\boxed{
\operatorname{logit}
P(Y_t=1)
=
a
+
b\ell_t.
}
```

Perfect calibration corresponds to

```math
\boxed{
a=0,
\qquad
b=1.
}
```

---

# 26. Calibration Intercept

If

```math
a>0
```

with slope otherwise appropriate, the model tends to underestimate the event probability.

If

```math
a<0,
```

the model tends to overestimate it.

Thus $a$ measures calibration-in-the-large.

---

# 27. Calibration Slope

If

```math
b<1,
```

predictions are generally too extreme.

If

```math
b>1,
```

predictions may be insufficiently dispersed.

The calibration slope is therefore a useful measure of probabilistic confidence quality.

---

# 28. Brier Score Decomposition

The Brier score can be decomposed into:

```math
\boxed{
BS
=
Reliability
-
Resolution
+
Uncertainty.
}
```

This decomposition separates three aspects of forecast quality.

---

# 29. Bin Notation

Let forecasts be partitioned into $K$ groups.

For group $k$:

```math
n_k
```

is group size,

```math
p_k
```

is average forecast probability,

```math
o_k
```

is observed event frequency.

Let overall event frequency be

```math
\bar o
=
\frac{1}{T}
\sum_{t=1}^{T}
Y_t.
```

---

# 30. Reliability Term

Define

```math
\boxed{
REL
=
\frac{1}{T}
\sum_{k=1}^{K}
n_k
(p_k-o_k)^2.
}
```

This measures miscalibration.

Perfectly calibrated bins satisfy

```math
p_k=o_k,
```

making

```math
REL=0.
```

Smaller reliability penalty is better.

---

# 31. Resolution Term

Define

```math
\boxed{
RES
=
\frac{1}{T}
\sum_{k=1}^{K}
n_k
(o_k-\bar o)^2.
}
```

Resolution measures whether forecast groups correspond to meaningfully different event frequencies.

Large resolution is good.

A model forecasting the same probability every time has approximately zero resolution.

---

# 32. Uncertainty Term

Define

```math
\boxed{
UNC
=
\bar o
(1-\bar o).
}
```

This is the unconditional variance of the binary outcome.

It is determined by the outcome process rather than the forecasting model.

---

# 33. Brier Decomposition Identity

The decomposition is

```math
\boxed{
BS
=
REL
-
RES
+
UNC.
}
```

Therefore good forecasts achieve:

```math
\text{low reliability error}
```

and

```math
\text{high resolution}.
```

---

# 34. Interpretation of Resolution

Suppose unconditional event probability is

```math
0.50.
```

A model always predicting

```math
0.50
```

can be perfectly calibrated.

But it provides no separation between different market states.

Its resolution is zero.

Thus:

```math
\boxed{
\text{Calibration alone is insufficient}.
}
```

A useful forecast should ideally be both calibrated and sharp enough to distinguish market conditions.

---

# 35. Sharpness

Sharpness refers to concentration of predictive probabilities away from the unconditional baseline.

A model producing only values near

```math
0.50
```

is not sharp.

A model producing

```math
0.20,
0.80,
0.35,
0.90
```

is sharper.

But sharpness is desirable only when calibration is retained.

---

# 36. Calibration-Sharpness Principle

The objective can be stated as:

```math
\boxed{
\text{maximize useful sharpness subject to calibration}.
}
```

Overconfident probabilities without calibration are not superior forecasts.

---

# 37. Baseline Probability Forecast

Let unconditional event frequency estimated from training data be

```math
\pi.
```

The simplest probability benchmark is

```math
\boxed{
p_t=\pi.
}
```

Its log loss provides a no-information benchmark.

Any sophisticated model should outperform this baseline before claims of predictive information are considered.

---

# 38. Entropy Baseline

If the true unconditional event probability is $\pi$, the optimal constant log loss is

```math
\boxed{
H(Y)
=
-\pi\log\pi
-
(1-\pi)\log(1-\pi).
}
```

Adding feature information $X$ can reduce optimal expected log loss to

```math
H(Y\mid X).
```

The difference is mutual information:

```math
\boxed{
I(Y;X)
=
H(Y)-H(Y\mid X).
}
```

---

# 39. Price Baseline Versus Information Model

For this project, the more important baseline is not only the unconditional forecast.

It is the price-only model:

```math
p_t^P.
```

The information-augmented probability is

```math
p_t^{I+P}.
```

Their observed log-loss difference is

```math
\boxed{
d_t
=
\ell_{\log}
(p_t^P,Y_t)
-
\ell_{\log}
(p_t^{I+P},Y_t).
}
```

This is the inferential quantity derived in the previous document.

---

# 40. Calibration Improvement Versus Predictive Improvement

Suppose a new information model lowers log loss.

The improvement may come from:

- better ranking,
- better probability magnitudes,
- better calibration,
- better resolution,
- combinations of these effects.

Therefore we should inspect not only loss reduction but also how forecast quality changed.

---

# 41. Recalibration

Suppose raw probability is

```math
p_t.
```

A calibration model transforms it into

```math
\boxed{
\tilde p_t
=
g(p_t).
}
```

The transformation should improve probability reliability without using future evaluation outcomes.

---

# 42. Platt Scaling

Let raw score be

```math
s_t.
```

Platt scaling fits

```math
\boxed{
\tilde p_t
=
\sigma
\left(
a+bs_t
\right).
}
```

If $s_t$ is the model's raw logit, this performs a parametric recalibration.

---

# 43. Logit Recalibration

If the model already outputs $p_t$, define

```math
s_t
=
\operatorname{logit}(p_t).
```

Then

```math
\boxed{
\operatorname{logit}
(\tilde p_t)
=
a
+
b
\operatorname{logit}(p_t).
}
```

This directly corrects calibration intercept and slope.

---

# 44. Temperature Scaling

A restricted form uses

```math
a=0.
```

Then

```math
\boxed{
\tilde p_t
=
\sigma
\left(
\frac{
\operatorname{logit}(p_t)
}{
T
}
\right).
}
```

If

```math
T>1,
```

extreme probabilities are softened.

---

# 45. Isotonic Regression

A nonparametric alternative estimates monotonic mapping

```math
\boxed{
\tilde p_t
=
g(p_t)
}
```

subject to $g$ being nondecreasing.

This allows flexible calibration.

But it can overfit small calibration samples.

---

# 46. Calibration Leakage

Suppose a model generates predictions on holdout data.

If the holdout outcomes are then used to fit

```math
g,
```

and the recalibrated probabilities are evaluated on the same holdout, the result is leaked.

Therefore calibration must itself be estimated using training or validation data only.

---

# 47. Chronological Calibration

For time-series prediction, the correct sequence is

```math
\boxed{
\text{Train Base Model}
\rightarrow
\text{Future Validation Predictions}
\rightarrow
\text{Fit Calibrator}
\rightarrow
\text{Later Evaluation Predictions}.
}
```

Random calibration splits should not break time ordering.

---

# 48. Cross-Fitted Calibration

Within development data, generate out-of-fold probabilities

```math
p_t^{OOF}.
```

Fit the calibration mapping using these genuinely out-of-sample probabilities.

This avoids allowing the calibrator to learn from overfitted in-sample predictions.

---

# 49. Calibrate Before or After Fusion?

Two architectures are possible.

### Architecture A

```math
\text{Specialist}
\rightarrow
\text{Specialist Calibration}
\rightarrow
\text{Fusion}.
```

### Architecture B

```math
\text{Specialists}
\rightarrow
\text{Fusion}
\rightarrow
\text{Final Calibration}.
```

A hybrid approach can calibrate specialists lightly and also test final-model calibration.

The choice should be empirical.

---

# 50. Forecast Reliability Across Regimes

A model may be globally calibrated but locally miscalibrated.

For regime $S_t=j$, examine

```math
\boxed{
E[Y_t\mid p_t,S_t=j].
}
```

We may have

```math
E[Y\mid p]\approx p
```

globally while

```math
E[Y\mid p,S=1]\neq p.
```

This motivates conditional calibration diagnostics.

---

# 51. Horizon-Specific Calibration

Calibration should be evaluated separately for each forecast horizon.

A model may be well calibrated at

```math
h=5\text{ minutes}
```

but overconfident at

```math
h=1\text{ hour}.
```

Therefore each primary horizon should have its own calibration analysis.

---

# 52. Time-Varying Base Rate

Suppose event probability changes through time:

```math
P(Y_t=1)
=
\pi_t.
```

A globally calibrated model may become temporarily miscalibrated if the base rate shifts.

This suggests monitoring calibration through rolling windows.

---

# 53. Calibration Drift

Define rolling calibration intercept and slope:

```math
a_t^{(w)},
\qquad
b_t^{(w)}.
```

Large persistent deviations from

```math
a=0,
\qquad
b=1
```

can signal model degradation or regime change.

---

# 54. Probability Calibration and Trading Decisions

Suppose a binary contract pays

```math
1
```

if the event occurs and

```math
0
```

otherwise.

If its market price is

```math
m,
```

the market-implied probability under simplifying assumptions is approximately

```math
m.
```

Our model estimates

```math
p.
```

Expected payoff from buying one contract at price $m$ is

```math
\boxed{
E[\Pi_{long}]
=
p-m.
}
```

Thus the economic decision depends directly on probability calibration.

---

# 55. Why Miscalibration Is Economically Dangerous

Suppose true probability is

```math
0.55,
```

but the model reports

```math
0.70.
```

A contract priced at

```math
0.62
```

appears attractive under the model:

```math
0.70-0.62=0.08.
```

But its true expected value is

```math
0.55-0.62=-0.07.
```

Thus overconfidence can reverse the sign of an economic decision.

---

# 56. Decision Threshold with Costs

Suppose entering a positive-direction trade yields gain $G$ when correct and loss $L$ when incorrect.

Expected payoff is

```math
E[\Pi]
=
pG
-
(1-p)L.
```

Trade only if

```math
pG
-
(1-p)L
>
0.
```

Therefore

```math
pG
>
L-pL.
```

Hence

```math
p(G+L)
>
L.
```

So the break-even probability is

```math
\boxed{
p^*
=
\frac{L}{G+L}.
}
```

The optimal probability threshold is not automatically $0.5$.

---

# 57. Including Transaction Costs

Let round-trip cost be $c$.

Then expected payoff becomes

```math
E[\Pi]
=
pG
-
(1-p)L
-
c.
```

Trade when

```math
p(G+L)
>
L+c.
```

Therefore

```math
\boxed{
p^*
=
\frac{
L+c
}{
G+L
}.
}
```

Higher transaction costs require stronger forecast confidence.

---

# 58. Symmetric Payoff Case

If

```math
G=L,
```

and transaction cost is zero,

```math
p^*
=
\frac{1}{2}.
```

With positive costs,

```math
p^*
>
\frac{1}{2}.
```

Thus a probability of

```math
0.51
```

may be statistically directional but economically useless.

---

# 59. Expected Value Versus Probability

High probability does not necessarily imply high expected value.

Suppose trade outcomes have unequal magnitudes.

Expected return is

```math
\boxed{
E[R]
=
pE[R\mid Y=1]
+
(1-p)
E[R\mid Y=0].
}
```

Therefore directional probability and expected-return modelling are related but distinct.

---

# 60. Calibration and Position Size

Suppose estimated edge is

```math
e_t
=
p_t-p_t^{break-even}.
```

A simple position-sizing rule might be

```math
w_t
=
k e_t
```

subject to risk limits.

If $p_t$ is overconfident, then position size becomes systematically excessive.

Thus calibration affects not only trade selection but also risk allocation.

---

# 61. Kelly Criterion Example

For a binary bet with net odds $b$ and win probability $p$, the Kelly fraction is

```math
\boxed{
f^*
=
\frac{
bp-(1-p)
}{
b
}.
}
```

Equivalently,

```math
f^*
=
\frac{
p(b+1)-1
}{
b
}.
```

The position depends directly on probability quality.

Probability estimation error therefore translates into capital-allocation error.

---

# 62. Prediction-Market Interpretation

For a binary prediction-market contract priced at $m$, ignoring fees, frictions and risk premia:

```math
\boxed{
EV_{long}
=
p-m.
}
```

For a short position,

```math
\boxed{
EV_{short}
=
m-p.
}
```

The model should therefore trade only when estimated edge exceeds:

- fees,
- spread,
- execution uncertainty,
- model uncertainty.

---

# 63. Probability Edge

Define

```math
\boxed{
Edge_t
=
p_t-m_t.
}
```

But the actionable quantity should include uncertainty.

For example,

```math
AdjustedEdge_t
=
Edge_t
-
\lambda U_t,
```

where $U_t$ represents forecast uncertainty.

This becomes relevant in the later economic-translation phase.

---

# 64. Calibration Under Distribution Shift

Suppose a model was trained in environment

```math
P_{train}(X,Y)
```

but deployed under

```math
P_{test}(X,Y).
```

Even if calibration was good in training, it may fail under distribution shift.

This is especially important in Bitcoin because:

- volatility changes,
- liquidity changes,
- venue structure evolves,
- regulatory regimes change.

---

# 65. Base-Rate Shift

Suppose

```math
P_{train}(Y=1)
\neq
P_{test}(Y=1).
```

Even stable ranking performance can coexist with calibration deterioration.

Therefore calibration must be monitored chronologically rather than treated as a permanent model property.

---

# 66. Conditional Calibration

A stronger requirement is

```math
\boxed{
E[Y\mid p,Z]
=
p,
}
```

for relevant conditioning variables $Z$.

Examples include:

```math
Z
=
\text{volatility regime},
```

```math
Z
=
\text{liquidity regime},
```

or

```math
Z
=
\text{forecast horizon}.
```

This is much stronger than global calibration.

---

# 67. Calibration and Specialist Fusion

Suppose specialist forecasts are

```math
p_O,
p_N,
p_C,
p_X.
```

If one specialist is systematically overconfident, logit stacking may learn a smaller coefficient for it.

However, this does not guarantee full calibration.

Therefore final fusion probabilities still require independent calibration evaluation.

---

# 68. Reliability of Specialist Probabilities

For each specialist $j$, estimate:

```math
REL_j,
```

```math
RES_j,
```

and

```math
BS_j.
```

A specialist with low Brier score may achieve this through strong resolution, strong calibration, or both.

This helps diagnose why particular information domains contribute value.

---

# 69. Calibration as Information Quality

Suppose a specialist repeatedly says

```math
p=0.80
```

but the event occurs only 55% of the time.

The specialist may contain directional signal but its stated information magnitude is unreliable.

Thus calibration provides a distinction between:

```math
\text{signal direction}
```

and

```math
\text{signal probability magnitude}.
```

---

# 70. Null Probability Model

Suppose the price-only model outputs

```math
p_t^P.
```

An information model should not receive credit simply for producing more extreme probabilities.

The information model must reduce proper scoring loss:

```math
\boxed{
\mathcal{L}_{proper}
\left(
p^{I+P}
\right)
<
\mathcal{L}_{proper}
\left(
p^P
\right).
}
```

This protects the research against confidence inflation.

---

# 71. Calibration Bootstrap

Calibration estimates also have sampling uncertainty.

Within each time-aware bootstrap sample, recompute quantities such as:

```math
REL,
```

```math
Brier,
```

or calibration slope.

This produces uncertainty intervals around calibration diagnostics.

The bootstrap procedure must preserve temporal dependence.

---

# 72. Reliability Diagram Uncertainty

Observed event frequency in a calibration bin follows sampling variation.

For bin $k$,

```math
\bar Y_k
=
\frac{1}{n_k}
\sum_{i\in B_k}Y_i.
```

Its approximate variance under independent Bernoulli sampling would be

```math
\frac{
q_k(1-q_k)
}{
n_k
}.
```

Financial dependence complicates this.

Therefore visual bin differences should not be overinterpreted without uncertainty.

---

# 73. Calibration Sample Size

Extreme probability bins often contain few observations.

For example, forecasts above

```math
0.90
```

may be rare.

A reliability diagram can therefore appear unstable in those regions.

This is another reason log loss and Brier score remain primary quantitative metrics.

---

# 74. Proper Scores and Model Selection

Hyperparameters should be selected using the same type of probabilistic objective that matters scientifically.

If final evaluation uses log loss, tuning exclusively for classification accuracy can produce a model that is poorly aligned with the research objective.

Thus:

```math
\boxed{
\text{Model Selection Metric}
\approx
\text{Final Scientific Metric}.
}
```

---

# 75. Calibration and Threshold Selection Must Be Separate

First estimate a probability model.

Then, if needed, apply an economically motivated decision threshold.

Do not distort the probability model simply to optimize a particular trade threshold.

The separation is:

```math
\boxed{
\text{Probability Estimation}
\neq
\text{Decision Policy}.
}
```

---

# 76. Forecasting Versus Decision Theory

Probability forecasting asks:

```math
P(Y=1\mid\mathcal{F}_t)=?
```

Decision theory asks:

```math
\arg\max_a
E[
U(a,Y)
\mid
\mathcal{F}_t
].
```

A probability forecast is therefore an input to a decision problem.

It is not itself the final trading rule.

---

# 77. Optimal Action

Given actions $a\in\mathcal{A}$ and utility $U(a,Y)$,

```math
\boxed{
a_t^*
=
\arg\max_{a\in\mathcal{A}}
E
\left[
U(a,Y)
\mid
\mathcal{F}_t
\right].
}
```

The conditional probability forecast determines the expectation.

This is why probability calibration has economic significance.

---

# 78. Probability Uncertainty

The model may estimate

```math
p_t
```

but the estimate itself is uncertain.

Let

```math
Var(\hat p_t)
```

represent probability-estimation uncertainty.

A robust decision rule may require a margin above break-even rather than trading whenever

```math
\hat p_t>p^*.
```

For example,

```math
\boxed{
\hat p_t
-
z_\alpha
SE(\hat p_t)
>
p^*.
}
```

This is one possible conservative framework.

---

# 79. Forecast Entropy

Binary predictive entropy is

```math
\boxed{
H_t
=
-p_t\log p_t
-
(1-p_t)\log(1-p_t).
}
```

Entropy is maximal at

```math
p_t=0.5
```

and decreases as probability approaches zero or one.

Thus entropy measures predictive uncertainty.

---

# 80. Entropy Versus Confidence

Low entropy means the model is confident.

It does not imply the model is correct.

Therefore entropy should be interpreted jointly with calibration.

A poorly calibrated model can exhibit low entropy while being systematically wrong.

---

# 81. Information Gain Relative to Price Model

Define observation-level gain

```math
g_t
=
\ell_{\log}
(p_t^P,Y_t)
-
\ell_{\log}
(p_t^{I+P},Y_t).
```

Average gain is

```math
\boxed{
IG
=
\frac{1}{T}
\sum_tg_t.
}
```

Measured using natural logs, this is expressed in nats per forecast.

---

# 82. Information Gain in Bits

Convert nats to bits:

```math
\boxed{
IG_{bits}
=
\frac{IG}{\ln2}.
}
```

This creates an interpretable bridge between probabilistic model performance and information theory.

---

# 83. Calibration Does Not Create New Information

Suppose recalibration improves log loss.

This does not necessarily mean the base model discovered additional predictive variables.

Calibration transforms existing model scores into better probability estimates.

Therefore the paper should distinguish:

```math
\text{representation gain}
```

from

```math
\text{calibration gain}.
```

---

# 84. Decomposing Improvement

Let:

```math
L_{raw}
```

be raw model loss,

```math
L_{cal}
```

be calibrated model loss,

and

```math
L_P
```

be price-baseline loss.

Then:

```math
L_P-L_{raw}
```

measures raw model improvement,

while

```math
L_{raw}-L_{cal}
```

measures calibration improvement.

This helps identify where performance gains originate.

---

# 85. Calibration Protocol for the Project

For every primary predictive model, final evaluation should include:

1. log loss;
2. Brier score;
3. calibration intercept;
4. calibration slope;
5. reliability diagram;
6. resolution analysis;
7. probability distribution / sharpness;
8. comparison before and after calibration.

These diagnostics should be computed strictly out of sample.

---

# 86. Final Holdout Rule

The calibrator must be frozen before the locked holdout is opened.

On holdout data:

```math
\boxed{
\tilde p_t
=
g_{\text{frozen}}
(p_t).
}
```

No calibration coefficient may be modified using holdout outcomes.

---

# 87. Falsifiable Predictions

The calibration framework generates several testable requirements.

### Prediction 1

The information model should reduce proper scoring loss, not merely increase accuracy.

### Prediction 2

Probability forecasts should remain reasonably calibrated out of sample.

### Prediction 3

Calibration slope should not collapse far below one on holdout data.

### Prediction 4

Any high-confidence forecast region should exhibit corresponding empirical event frequencies.

### Prediction 5

Recalibration should not account for the entire apparent advantage of the information model if the underlying representation genuinely contains new information.

### Prediction 6

Decision rules based on probabilities should remain economically sensible after costs.

---

# 88. What Would Reject a Strong Probability Claim?

We should not describe a model as a strong probabilistic forecaster if:

- log loss does not improve,
- Brier score deteriorates materially,
- calibration slope indicates severe overconfidence,
- reliability diagrams reveal large systematic deviations,
- gains occur only after leaked recalibration,
- extreme probabilities are unsupported by observed frequencies,
- the model produces attractive classification accuracy but poor proper scoring performance.

---

# 89. Mathematical Core

### Log loss

```math
\boxed{
\ell_{\log}
=
-
\left[
Y\log p
+
(1-Y)\log(1-p)
\right]
}
```

### Strict propriety

```math
\boxed{
p^*
=
q
=
P(Y=1\mid X)
}
```

### Cross-entropy decomposition

```math
\boxed{
R_{\log}(p;q)
=
H(q)
+
D_{KL}(q\Vert p)
}
```

### Brier score

```math
\boxed{
BS=(p-Y)^2
}
```

### Brier decomposition

```math
\boxed{
BS
=
REL
-
RES
+
UNC
}
```

### Perfect calibration

```math
\boxed{
E[Y\mid p]
=
p
}
```

### Calibration model

```math
\boxed{
\operatorname{logit}
P(Y=1)
=
a
+
b
\operatorname{logit}(p)
}
```

with ideal values

```math
\boxed{
a=0,
\qquad
b=1.
}
```

### Break-even probability

```math
\boxed{
p^*
=
\frac{
L+c
}{
G+L
}
}
```

---

# 90. Connection to the Main Research Thesis

The complete probability layer is now:

```math
\boxed{
\text{Multimodal Information}
\rightarrow
\text{Probability Forecast}
\rightarrow
\text{Proper Score}
\rightarrow
\text{Calibration}
\rightarrow
\text{Decision Threshold}
\rightarrow
\text{Economic Action}.
}
```

The project therefore does not ask only:

> Did the model guess the direction correctly?

It asks:

> Did the information architecture produce a more accurate conditional probability distribution than Bitcoin price history alone?

Mathematically, the central comparison remains

```math
\boxed{
\mathcal{L}_{\log}
\left(
\mathcal{M}_{I+P}
\right)
<
\mathcal{L}_{\log}
\left(
\mathcal{M}_{P}
\right).
}
```

But the resulting probabilities must also be reliable enough that

```math
p_t
```

can be interpreted as genuine economic probabilities rather than arbitrary model scores.

That is the standard required before those probabilities can be used for trading, position sizing or prediction-market valuation.
