# Derivation IX — Statistical Inference for Predictive Information

## Dependent Forecast Errors, HAC Estimation and Block Bootstrap

The central empirical claim of the research programme is not merely that one model achieves a lower historical loss than another.

The relevant question is whether the observed improvement is sufficiently large and sufficiently stable that it is unlikely to be explained by sampling variation.

The principal comparison is

```math
\mathcal{M}_{P}
```

against

```math
\mathcal{M}_{I+P}.
```

Let the corresponding out-of-sample losses be

```math
\ell_{P,t}
```

and

```math
\ell_{I+P,t}.
```

The central inferential object is the loss differential

```math
\boxed{
d_t
=
\ell_{P,t}
-
\ell_{I+P,t}.
}
```

Positive $d_t$ means the information-augmented model performed better at observation $t$.

The mathematical path is

```math
\boxed{
\text{Loss Differential}
\rightarrow
\text{Serial Dependence}
\rightarrow
\text{Long-Run Variance}
\rightarrow
\text{HAC Estimation}
\rightarrow
\text{Block Resampling}
\rightarrow
\text{Confidence Interval}
\rightarrow
\text{Research Claim}.
}
```

---

# 1. Population Loss Difference

Define expected predictive loss for model $M$ as

```math
\mathcal{L}(M)
=
E[
\ell_{M,t}
].
```

For the price-only model,

```math
\mathcal{L}_P
=
E[
\ell_{P,t}
].
```

For the information-augmented model,

```math
\mathcal{L}_{I+P}
=
E[
\ell_{I+P,t}
].
```

Define population improvement

```math
\boxed{
\Delta
=
\mathcal{L}_P
-
\mathcal{L}_{I+P}.
}
```

If

```math
\Delta>0,
```

the information-augmented model has lower expected loss.

---

# 2. Sample Loss Difference

Given $T$ out-of-sample forecasts,

```math
\hat{\Delta}
=
\frac{1}{T}
\sum_{t=1}^{T}
d_t.
```

Therefore,

```math
\boxed{
\hat{\Delta}
=
\bar d.
}
```

This is the empirical estimate of incremental predictive gain.

---

# 3. Connection to Conditional Information

Under optimal logarithmic forecasts,

```math
I(Y;I\mid P)
=
\mathcal{L}_P^*
-
\mathcal{L}_{P,I}^*.
```

Thus the empirical statistic

```math
\hat{\Delta}_{log}
=
\bar d
```

can be interpreted as an out-of-sample estimate of the practical information gain captured by the fitted models.

This interpretation is approximate because the empirical models need not equal the population-optimal conditional distributions.

---

# 4. Hypothesis Test

A natural one-sided hypothesis is

```math
H_0:
\Delta\leq0
```

against

```math
H_1:
\Delta>0.
```

Equivalently,

```math
H_0:
E[d_t]\leq0
```

against

```math
H_1:
E[d_t]>0.
```

The challenge is estimating the sampling uncertainty of

```math
\bar d.
```

---

# 5. Why IID Standard Errors Are Inappropriate

If observations were independent,

```math
Var(\bar d)
=
\frac{
Var(d_t)
}{
T
}.
```

Then the standard error would be

```math
SE_{iid}
=
\sqrt{
\frac{
\hat\gamma_0
}{
T
}
},
```

where

```math
\gamma_0
=
Var(d_t).
```

Financial forecast losses, however, are generally dependent through time.

Therefore

```math
Cov(d_t,d_{t-k})
```

may be nonzero.

Ignoring this dependence usually understates uncertainty.

---

# 6. Sources of Dependence

Serial dependence may arise from:

- overlapping forecast horizons,
- persistent volatility,
- persistent liquidity regimes,
- clustered news events,
- repeated retraining windows,
- autocorrelated prediction errors,
- common market shocks.

Thus the relevant variance is not merely the one-period variance.

---

# 7. Overlapping Forecast Horizons

Suppose predictions are generated every $\Delta$ time units for horizon $h$.

For example:

```math
\Delta
=
1\text{ minute}
```

and

```math
h
=
15\text{ minutes}.
```

Then adjacent outcomes

```math
R_{t,t+h}
```

and

```math
R_{t+\Delta,t+\Delta+h}
```

share most of the same future return interval.

This mechanically induces dependence.

---

# 8. Number of Overlapping Observations

Define

```math
m
=
\left\lceil
\frac{h}{\Delta}
\right\rceil.
```

Then a target can overlap approximately the next

```math
m-1
```

forecast origins.

Therefore a natural minimum dependence scale induced by label overlap is

```math
\boxed{
L_h
=
\left\lceil
\frac{h}{\Delta}
\right\rceil
-
1.
}
```

This does not mean all dependence disappears after $L_h$.

It provides only a structural lower bound for considering serial correlation.

---

# 9. Variance of a Dependent Sample Mean

Let

```math
\gamma_k
=
Cov(d_t,d_{t-k}).
```

Then

```math
Var
\left(
\sum_{t=1}^{T}d_t
\right)
=
\sum_{t=1}^{T}
\sum_{s=1}^{T}
Cov(d_t,d_s).
```

Under covariance stationarity,

```math
Var(\bar d)
=
\frac{1}{T^2}
\left[
T\gamma_0
+
2
\sum_{k=1}^{T-1}
(T-k)\gamma_k
\right].
```

Therefore,

```math
\boxed{
Var(\bar d)
=
\frac{1}{T}
\left[
\gamma_0
+
2
\sum_{k=1}^{T-1}
\left(
1-\frac{k}{T}
\right)
\gamma_k
\right].
}
```

---

# 10. Long-Run Variance

As $T$ becomes large, define the long-run variance

```math
\boxed{
\Omega
=
\gamma_0
+
2
\sum_{k=1}^{\infty}
\gamma_k.
}
```

Then asymptotically,

```math
\boxed{
Var(\bar d)
\approx
\frac{\Omega}{T}.
}
```

Thus,

```math
\boxed{
SE(\bar d)
\approx
\sqrt{
\frac{\Omega}{T}
}.
}
```

---

# 11. Spectral Interpretation

The long-run variance is related to the spectral density of $d_t$ at frequency zero.

If

```math
f_d(\omega)
```

is the spectral density, then under the common normalization

```math
\boxed{
\Omega
=
2\pi f_d(0).
}
```

Thus inference about the sample mean depends on very low-frequency variation in the loss-differential process.

---

# 12. Sample Autocovariance

Estimate lag-$k$ autocovariance using

```math
\hat\gamma_k
=
\frac{1}{T}
\sum_{t=k+1}^{T}
(d_t-\bar d)
(d_{t-k}-\bar d).
```

For $k=0$,

```math
\hat\gamma_0
=
\frac{1}{T}
\sum_{t=1}^{T}
(d_t-\bar d)^2.
```

A naive estimate of long-run variance would sum many autocovariances.

That estimate can become unstable at large lags.

---

# 13. Truncated Long-Run Variance

One approach truncates the autocovariance sum at lag $L$:

```math
\hat\Omega
=
\hat\gamma_0
+
2
\sum_{k=1}^{L}
\hat\gamma_k.
```

But sharp truncation gives equal weight to every included lag and zero weight immediately afterward.

A smoother weighting scheme is usually preferable.

---

# 14. HAC Estimator

A heteroskedasticity-and-autocorrelation-consistent estimator takes the general form

```math
\boxed{
\hat\Omega_{HAC}
=
\hat\gamma_0
+
2
\sum_{k=1}^{L}
w_k
\hat\gamma_k.
}
```

The weights satisfy approximately

```math
0\leq w_k\leq1.
```

Different kernels produce different HAC estimators.

---

# 15. Newey–West Bartlett Weights

The Newey–West estimator commonly uses Bartlett weights:

```math
\boxed{
w_k
=
1
-
\frac{k}{L+1}.
}
```

Thus,

```math
\boxed{
\hat\Omega_{NW}
=
\hat\gamma_0
+
2
\sum_{k=1}^{L}
\left(
1-\frac{k}{L+1}
\right)
\hat\gamma_k.
}
```

More distant autocovariances receive lower weight.

---

# 16. Why the Bartlett Kernel Helps

Sample autocovariances become increasingly noisy at large lags.

The Bartlett weights gradually shrink higher-lag contributions toward zero.

This produces a more stable long-run variance estimate than treating all lags equally.

---

# 17. HAC Standard Error

Once

```math
\hat\Omega_{NW}
```

is obtained,

```math
\boxed{
\widehat{SE}_{HAC}
\left(
\bar d
\right)
=
\sqrt{
\frac{
\hat\Omega_{NW}
}{
T
}
}.
}
```

A corresponding test statistic is

```math
\boxed{
T_{HAC}
=
\frac{
\bar d
}{
\sqrt{
\hat\Omega_{NW}/T
}
}.
}
```

---

# 18. Lag-Length Choice

The HAC lag $L$ cannot be selected merely to maximize significance.

Possible considerations include:

- forecast overlap,
- empirical autocorrelation,
- sample size,
- automatic bandwidth rules.

For overlapping horizons, a reasonable minimum diagnostic scale is

```math
L
\geq
L_h.
```

But longer persistence in losses may require a larger bandwidth.

The final rule must be fixed before final holdout inference.

---

# 19. Positive Serial Correlation Effect

Suppose

```math
\gamma_k>0
```

for several lags.

Then

```math
\Omega
>
\gamma_0.
```

Therefore

```math
SE_{HAC}
>
SE_{iid}.
```

This illustrates why treating overlapping predictions as independent can create falsely narrow confidence intervals.

---

# 20. Negative Serial Correlation

If some autocovariances are negative,

```math
\Omega
```

can be smaller than

```math
\gamma_0.
```

Thus dependence does not mechanically increase standard errors.

The relevant object is the entire long-run covariance structure.

---

# 21. Forecast Comparison Statistic

A generic forecast-loss comparison statistic is

```math
\boxed{
DM
=
\frac{
\bar d
}{
\sqrt{
\hat\Omega/T
}
}.
}
```

This resembles the Diebold–Mariano forecast-comparison framework.

Its validity depends on the assumptions of the particular forecasting design.

---

# 22. Nested-Model Complication

Our principal comparison may involve nested models.

For example,

```math
\mathcal{M}_P
```

may be nested within

```math
\mathcal{M}_{P+I}.
```

Under the null that the additional information has no predictive value, parameter estimation in the larger model can create non-standard finite-sample forecast-comparison behavior.

Therefore we should not treat a generic asymptotic $DM$ statistic as the sole inferential result.

This motivates bootstrap inference and robustness checks.

---

# 23. Statistical Versus Economic Significance

Suppose

```math
\bar d
>
0
```

and the confidence interval excludes zero.

This establishes statistical evidence of lower forecast loss.

It does not automatically imply:

```math
\text{profitable trading}.
```

Economic relevance additionally depends on:

- size of the gain,
- transaction costs,
- probability calibration,
- turnover,
- capital constraints.

The two claims must remain distinct.

---

# 24. Confidence Interval from HAC

A large-sample two-sided interval is

```math
\boxed{
\bar d
\pm
z_{1-\alpha/2}
\sqrt{
\frac{
\hat\Omega
}{
T
}
}.
}
```

For a 95% interval,

```math
z_{0.975}
\approx
1.96.
```

Evidence that the information model has lower loss is strongest when the lower bound of the improvement interval remains above zero.

---

# 25. Equivalent Loss-Difference Convention

Some documents may instead define

```math
\delta_t
=
\ell_{I+P,t}
-
\ell_{P,t}.
```

Then improvement corresponds to

```math
E[\delta_t]<0.
```

Both conventions are valid.

This project should use one sign convention consistently.

We adopt

```math
\boxed{
d_t
=
\ell_{P,t}
-
\ell_{I+P,t},
}
```

so that

```math
d_t>0
```

means information improves the forecast.

---

# 26. Why Bootstrap?

Analytical HAC inference relies on asymptotic approximations.

Financial forecast losses may exhibit:

- heavy tails,
- nonlinear dependence,
- heteroskedasticity,
- regime shifts,
- finite-sample effects.

A time-aware bootstrap approximates the sampling distribution while preserving important serial dependence.

---

# 27. IID Bootstrap Failure

An ordinary bootstrap resamples individual observations independently:

```math
d_1^*,\ldots,d_T^*
```

from the empirical distribution of the original observations.

This destroys the temporal ordering and autocorrelation structure.

Therefore it is inappropriate when serial dependence is material.

---

# 28. Moving-Block Bootstrap

Choose block length $L$.

Construct overlapping blocks:

```math
B_1
=
(d_1,\ldots,d_L),
```

```math
B_2
=
(d_2,\ldots,d_{L+1}),
```

and so on until

```math
B_{T-L+1}.
```

There are

```math
T-L+1
```

possible overlapping blocks.

---

# 29. Bootstrap Resample Construction

Randomly sample blocks with replacement:

```math
B_{i_1},
B_{i_2},
\ldots,
B_{i_m}.
```

Concatenate them until at least $T$ observations are obtained.

Then truncate to length $T$.

This gives bootstrap sequence

```math
d_1^*,\ldots,d_T^*.
```

Within each selected block, the original dependence structure is preserved.

---

# 30. Bootstrap Mean

For bootstrap replication $b$, compute

```math
\boxed{
\bar d^{*(b)}
=
\frac{1}{T}
\sum_{t=1}^{T}
d_t^{*(b)}.
}
```

Repeat for

```math
b=1,\ldots,B.
```

This produces empirical bootstrap distribution

```math
\{
\bar d^{*(1)},
\ldots,
\bar d^{*(B)}
\}.
```

---

# 31. Percentile Bootstrap Interval

Let

```math
q_{\alpha/2}
```

and

```math
q_{1-\alpha/2}
```

denote empirical bootstrap quantiles.

Then a percentile interval is

```math
\boxed{
CI_{1-\alpha}^{perc}
=
\left[
q_{\alpha/2},
q_{1-\alpha/2}
\right].
}
```

This is easy to compute, though not always the most accurate bootstrap interval.

---

# 32. Basic Bootstrap Interval

Let the bootstrap distribution describe

```math
\bar d^*-\bar d.
```

A basic bootstrap interval is

```math
\boxed{
\left[
2\bar d-q_{1-\alpha/2},
\,
2\bar d-q_{\alpha/2}
\right].
}
```

This reflects the bootstrap error distribution around the observed estimate.

---

# 33. Studentized Bootstrap

A studentized statistic is

```math
T^*
=
\frac{
\bar d^*-\bar d
}{
\widehat{SE}^*
}.
```

Bootstrap quantiles of $T^*$ can then be used to construct confidence intervals.

Studentization can improve accuracy but requires estimating uncertainty within each bootstrap replication.

This increases computational cost substantially.

---

# 34. Block-Length Trade-Off

If $L$ is too small:

```math
\text{important dependence is destroyed}.
```

If $L$ is too large:

```math
\text{few effectively independent blocks remain}.
```

Thus block length controls a bias-variance trade-off.

It must not be tuned to produce favorable significance.

---

# 35. Forecast-Horizon Lower Bound

Because overlapping targets produce at least approximately

```math
L_h
=
\left\lceil
\frac{h}{\Delta}
\right\rceil
-
1
```

lags of mechanical dependence, the bootstrap block should not be chosen blindly below the dependence scale implied by the label construction.

Empirical dependence may require still longer blocks.

---

# 36. Circular Block Bootstrap

A practical variant treats the time series as circular.

After $d_T$, the sequence wraps to $d_1$.

Then every observation can serve as the start of a block.

This avoids giving central observations a higher probability of appearing within complete blocks than boundary observations.

---

# 37. Stationary Bootstrap

The stationary bootstrap uses blocks of random length.

At each step:

- with probability $1-p$, continue to the next chronological observation,
- with probability $p$, start a new block at a randomly selected observation.

Block lengths therefore follow a geometric distribution.

---

# 38. Expected Stationary-Bootstrap Block Length

If block termination probability is $p$, then

```math
P(L=\ell)
=
(1-p)^{\ell-1}p.
```

This is geometric.

Its expected value is

```math
\boxed{
E[L]
=
\frac{1}{p}.
}
```

Thus selecting $p$ is equivalent to selecting the expected dependence-preserving block length.

---

# 39. Why Random Block Lengths Help

Fixed block boundaries can introduce artificial periodic structure.

Random block lengths reduce this artifact.

The stationary bootstrap also produces an approximately stationary resampled sequence when applied appropriately.

It is therefore an attractive primary candidate for financial forecast-loss inference.

---

# 40. Stationary-Bootstrap Algorithm

Start at a randomly selected original index.

Copy the corresponding observation.

For each next bootstrap observation:

with probability

```math
1-p,
```

advance to the next chronological observation;

with probability

```math
p,
```

jump to a new randomly selected starting index.

Wrap around the sample when necessary.

Continue until $T$ bootstrap observations are generated.

---

# 41. Bootstrap Under the Null

For hypothesis testing, it may be desirable to construct the bootstrap distribution under

```math
H_0:
E[d_t]=0.
```

Define centered losses:

```math
\tilde d_t
=
d_t-\bar d.
```

Resample blocks of

```math
\tilde d_t
```

rather than raw $d_t$.

Then each bootstrap replication approximates the null distribution of the sample mean.

---

# 42. Null Bootstrap Statistic

For replication $b$,

```math
\bar{\tilde d}^{*(b)}
=
\frac{1}{T}
\sum_t
\tilde d_t^{*(b)}.
```

The one-sided bootstrap $p$-value can be approximated by

```math
\boxed{
\hat p
=
\frac{
1+
\sum_{b=1}^{B}
\mathbf{1}
\left[
\bar{\tilde d}^{*(b)}
\geq
\bar d
\right]
}{
B+1
}.
}
```

The exact direction must match the chosen sign convention.

---

# 43. One-Sided Inference

Because our primary scientific alternative is

```math
\Delta>0,
```

a one-sided test is scientifically meaningful if specified before examining the holdout.

The one-sided 95% lower confidence bound is especially useful.

If

```math
LB_{95\%}>0,
```

then the data support positive incremental predictive gain under the chosen inference procedure.

---

# 44. Fold-Level Dependence

Suppose predictions are generated through $K$ walk-forward folds.

Losses within each fold may be serially dependent.

Performance across folds may also differ because market regimes change.

Therefore we should retain:

```math
\bar d_1,\ldots,\bar d_K
```

rather than reporting only the pooled average.

---

# 45. Fold-Specific Gain

For fold $k$,

```math
\boxed{
\bar d_k
=
\frac{
1
}{
T_k
}
\sum_{t\in\mathcal{E}_k}
d_t.
}
```

The set

```math
\{
\bar d_1,\ldots,\bar d_K
\}
```

reveals whether the model improvement is distributed through time.

---

# 46. Pooled Gain

The overall sample gain is

```math
\bar d
=
\frac{
\sum_k
T_k\bar d_k
}{
\sum_kT_k
}.
```

Thus large evaluation folds naturally receive greater weight.

But the pooled estimate may conceal instability across periods.

---

# 47. Sign Stability

A simple descriptive stability statistic is

```math
\boxed{
S
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbf{1}
\{
\bar d_k>0
\}.
}
```

This is the proportion of evaluation folds in which the information model outperforms the price benchmark.

It is a diagnostic, not a replacement for formal inference.

---

# 48. Cumulative Loss Differential

Define

```math
\boxed{
C_t
=
\sum_{s=1}^{t}
d_s.
}
```

If the information model produces persistent gain,

```math
C_t
```

should tend to rise through time.

Flat or sharply reversing regions identify periods where the apparent information advantage disappears.

---

# 49. Interpretation of the Cumulative Curve

A smoothly rising curve suggests broadly distributed improvement.

A single large jump suggests performance concentrated in one episode.

A rise followed by persistent decline may indicate signal decay or structural change.

This figure should become an important final-paper diagnostic.

---

# 50. Rolling Mean Gain

Define a rolling window of length $w$:

```math
\boxed{
\bar d_t^{(w)}
=
\frac{1}{w}
\sum_{j=0}^{w-1}
d_{t-j}.
}
```

This gives a local estimate of forecast advantage.

It is useful for studying regime dependence and temporal degradation.

It should not be used to cherry-pick favorable subperiods.

---

# 51. Heteroskedastic Loss Differentials

Financial volatility changes through time.

Therefore

```math
Var(d_t)
```

may be state dependent.

HAC estimators permit heteroskedasticity, which is one reason they are appropriate here.

The block bootstrap also preserves clusters of high- and low-variance observations within sampled blocks.

---

# 52. Regime-Conditional Inference

If the model later studies regime-specific information value, define

```math
d_t^{(j)}
```

for observations associated probabilistically or explicitly with regime $j$.

Then estimate

```math
\Delta_j
=
E[d_t\mid S_t=j].
```

Inference becomes more difficult because regime classification is estimated rather than observed.

Such results should therefore be secondary to the global primary hypothesis.

---

# 53. Domain-Level Loss Differential

For information domain $j$, define

```math
d_{j,t}
=
\ell_{P,t}
-
\ell_{P+j,t}.
```

Then

```math
\boxed{
\Delta_j
=
E[d_{j,t}]
}
```

measures the incremental predictive value of that domain relative to the price benchmark.

The same HAC and bootstrap machinery can be applied to each primary domain comparison.

---

# 54. Ablation Loss Differential

For the full multimodal model and a model excluding domain $j$,

```math
a_{j,t}
=
\ell_{-j,t}
-
\ell_{full,t}.
```

Then

```math
E[a_{j,t}]>0
```

means the domain improves the full model.

This provides inferential uncertainty around domain ablation rather than reporting only point estimates.

---

# 55. Multiple Horizons

Suppose primary horizons are

```math
h_1,\ldots,h_H.
```

Then each produces hypothesis

```math
H_{0,h}:
\Delta_h\leq0.
```

Testing many horizons increases false-discovery risk.

Therefore the final paper should distinguish:

- primary horizons,
- secondary horizons,
- exploratory horizons.

---

# 56. Multiple Domains

Likewise, testing

```math
O,
N,
C,
X,
V
```

creates several domain-level hypotheses.

If every possible source and horizon is tested and only significant results are reported, nominal confidence levels become misleading.

This is the statistical reason the project predefines a small primary hypothesis family.

---

# 57. Family-Wise Versus False-Discovery Control

If strong simultaneous control is required, one may use family-wise procedures.

If many exploratory but related hypotheses are tested, false-discovery-rate control may be more appropriate.

The exact method depends on the final hypothesis structure.

This decision should be made before holdout analysis.

---

# 58. Selection Bias from Model Search

Suppose $M$ candidate models are evaluated and we report

```math
\min_m
\hat{\mathcal{L}}_m.
```

Even if all models have identical population performance, the selected minimum tends to be optimistically biased.

This is an order-statistics effect.

Therefore final inference cannot pretend that the selected model was specified without prior search.

---

# 59. Research Log as Statistical Control

Recording all material experiments allows us to quantify the scope of specification search.

This is not merely project management.

It is part of the statistical evidence because it documents:

```math
\text{researcher degrees of freedom}.
```

The experiment log should therefore remain version controlled.

---

# 60. Development Versus Confirmatory Evidence

Results obtained during model development are useful for discovery.

They should not receive the same evidentiary weight as the locked holdout.

The project therefore separates:

```math
\boxed{
\text{Exploratory Evidence}
}
```

from

```math
\boxed{
\text{Confirmatory Holdout Evidence}.
}
```

---

# 61. Locked-Holdout Loss Difference

After the complete research design is frozen, let holdout losses be

```math
d_1^H,\ldots,d_{T_H}^H.
```

Then

```math
\boxed{
\hat\Delta_H
=
\frac{1}{T_H}
\sum_{t=1}^{T_H}
d_t^H.
}
```

This becomes the most important empirical estimate in the project.

---

# 62. Holdout Confidence Interval

Using a preselected block-bootstrap procedure, construct

```math
CI_{95\%}
\left(
\Delta_H
\right).
```

The strongest evidence for the central hypothesis occurs if

```math
\boxed{
LB_{95\%}
\left(
\Delta_H
\right)
>0.
}
```

This means the entire confidence interval supports lower expected loss from the information-augmented model.

---

# 63. Weak Positive Result

Suppose

```math
\hat\Delta_H>0
```

but the interval contains zero.

Then the correct conclusion is:

> The point estimate favors the information model, but uncertainty remains too large to establish a reliable improvement at the chosen confidence level.

It should not be reported as a confirmed predictive edge.

---

# 64. Null Result

Suppose

```math
\hat\Delta_H\approx0.
```

Then the evidence suggests that the tested information architecture did not produce material incremental predictive gain over the price benchmark.

This remains scientifically meaningful.

---

# 65. Negative Result

Suppose

```math
\hat\Delta_H<0.
```

Then the information-augmented model performs worse than the price-only model on the holdout.

Possible explanations include:

- no incremental information,
- noisy information channels,
- model estimation error,
- structural instability,
- excessive dimensionality.

The negative result should not be optimized away.

---

# 66. Effect Size

Statistical significance alone is insufficient.

Define relative log-loss improvement:

```math
\boxed{
RI
=
\frac{
\mathcal{L}_P
-
\mathcal{L}_{I+P}
}{
\mathcal{L}_P
}.
}
```

Empirically,

```math
\hat{RI}
=
\frac{
\hat{\mathcal{L}}_P
-
\hat{\mathcal{L}}_{I+P}
}{
\hat{\mathcal{L}}_P
}.
```

This expresses improvement as a fraction of baseline loss.

---

# 67. Information Gain in Nats

With natural logarithmic loss,

```math
\hat\Delta_{log}
```

is measured in nats per prediction.

Thus,

```math
\boxed{
IG_{nats}
=
\hat\Delta_{log}.
}
```

To convert to bits,

```math
\boxed{
IG_{bits}
=
\frac{
\hat\Delta_{log}
}{
\ln2
}.
}
```

This gives an information-theoretic effect size.

---

# 68. Total Information Gain

If average gain is

```math
IG
```

nats per forecast across $T$ forecasts, cumulative predictive information gain is

```math
\boxed{
T\times IG.
}
```

This quantity must be interpreted carefully because observations can be dependent.

It is descriptive, not an independent-sample significance measure.

---

# 69. Economic Translation Comes Later

The inferential chain is:

```math
\boxed{
\text{Forecast Gain}
\rightarrow
\text{Statistical Confidence}
\rightarrow
\text{Calibration}
\rightarrow
\text{Trading Rule}
\rightarrow
\text{Costs}
\rightarrow
\text{Economic Value}.
}
```

Trading performance cannot substitute for forecast inference.

---

# 70. Recommended Primary Inference Architecture

For the final project, the default primary inference plan should be:

1. calculate observation-level out-of-sample log-loss differentials;
2. report mean gain;
3. report fold-specific gains;
4. report cumulative gain;
5. estimate HAC uncertainty as an analytical robustness check;
6. use a preselected block bootstrap as the principal finite-sample inference procedure;
7. report a confidence interval for the gain;
8. apply the same framework to predefined domain ablations.

This provides several complementary views of the same research quantity.

---

# 71. Why We Should Not Depend on One Test

No single inferential method is perfect.

HAC relies on asymptotic covariance estimation.

Block bootstrap relies on block-length choice and approximate local dependence.

Fold stability is descriptive.

Holdout performance is finite sample.

A persuasive result should therefore survive several reasonable representations of uncertainty rather than one favorable statistic.

---

# 72. Robustness Matrix

For primary comparison, final reporting should eventually contain something like:

| Evidence | Result |
|---|---|
| Mean log-loss gain | estimate |
| HAC interval | interval |
| Block-bootstrap interval | interval |
| Positive walk-forward folds | proportion |
| Holdout gain | estimate |
| Calibration change | estimate |
| Relative loss improvement | estimate |

No one row alone determines the conclusion.

---

# 73. Point-in-Time Requirement for Inference

Inference is only meaningful if the forecasts themselves are valid.

Therefore every loss entering

```math
d_t
```

must come from a model whose:

- features were point-in-time,
- preprocessing used training data only,
- hyperparameters were selected without future observations,
- predictions were genuinely out of sample.

Statistical sophistication cannot repair leaked forecasts.

---

# 74. Falsifiable Predictions

The inference framework creates concrete requirements.

### Prediction 1

If non-price information is useful,

```math
\bar d>0.
```

### Prediction 2

The gain should not depend entirely on one evaluation block.

### Prediction 3

HAC and block-bootstrap intervals should not contradict the claim of strong improvement.

### Prediction 4

The locked-holdout estimate should be directionally consistent with development evidence.

### Prediction 5

Domain-level improvements should survive uncertainty analysis if those domains genuinely contribute unique information.

---

# 75. What Would Reject the Strong Claim?

The strong incremental-information claim should not be made if:

- the mean gain is negative,
- the holdout reverses development results,
- confidence intervals include economically important negative values,
- improvement occurs only in one short episode,
- significance disappears after correcting dependence,
- results depend on one arbitrary bootstrap block length,
- the selected model reflects extensive unreported search.

---

# 76. Mathematical Core

The main inferential equations are:

### Loss differential

```math
\boxed{
d_t
=
\ell_{P,t}
-
\ell_{I+P,t}
}
```

### Mean gain

```math
\boxed{
\hat\Delta
=
\bar d
=
\frac{1}{T}
\sum_t d_t
}
```

### Long-run variance

```math
\boxed{
\Omega
=
\gamma_0
+
2
\sum_{k=1}^{\infty}
\gamma_k
}
```

### Newey–West estimator

```math
\boxed{
\hat\Omega_{NW}
=
\hat\gamma_0
+
2
\sum_{k=1}^{L}
\left(
1-\frac{k}{L+1}
\right)
\hat\gamma_k
}
```

### HAC standard error

```math
\boxed{
SE_{HAC}
=
\sqrt{
\frac{
\hat\Omega_{NW}
}{
T
}
}
}
```

### Moving-block bootstrap statistic

```math
\boxed{
\bar d^{*(b)}
=
\frac{1}{T}
\sum_t
d_t^{*(b)}
}
```

### Stationary-bootstrap expected block length

```math
\boxed{
E[L]
=
\frac{1}{p}
}
```

### Information gain in bits

```math
\boxed{
IG_{bits}
=
\frac{
\hat\Delta_{log}
}{
\ln2
}
}
```

---

# 77. Connection to the Main Research Thesis

The project can now move from:

```math
\text{Model A has lower loss than Model B}
```

to the substantially stronger statement:

```math
\boxed{
\text{The information-augmented model has lower expected predictive loss,}
}
```

subject to a quantified uncertainty interval that respects temporal dependence.

The complete inference chain is:

```math
\boxed{
\text{Point-in-Time Forecasts}
\rightarrow
\text{Observation-Level Losses}
\rightarrow
\text{Loss Differential}
\rightarrow
\text{Dependence-Aware Uncertainty}
\rightarrow
\text{Confidence Interval}
\rightarrow
\text{Scientific Conclusion}.
}
```

This is the mathematical standard required before the project can claim that observable information contributes genuine incremental predictive value.
