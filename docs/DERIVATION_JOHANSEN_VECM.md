# Derivation V — Johansen Cointegration, VECM and Multivariate Price Discovery

## From Multiple Stochastic Trends to a Reduced-Rank Information System

The previous derivations established two ideas.

First, individually non-stationary financial prices may share stationary equilibrium relations.

Second, those equilibrium deviations can enter short-run return dynamics through an error-correction mechanism.

A two-variable Engle–Granger system is useful for intuition, but the research programme is explicitly multivariate.

Bitcoin may simultaneously interact with:

- Ethereum,
- dollar conditions,
- interest rates,
- equities,
- gold,
- volatility,
- liquidity variables.

We therefore require a framework that estimates:

```math
\text{cointegration rank},
```

```math
\text{long-run equilibrium vectors},
```

and

```math
\text{adjustment coefficients}
```

jointly.

The Johansen framework provides this structure.

The mathematical path is:

```math
\boxed{
VAR
\rightarrow
VECM
\rightarrow
\Pi
\rightarrow
\operatorname{rank}(\Pi)
\rightarrow
\alpha\beta^\top
\rightarrow
\text{Reduced-Rank Likelihood}
\rightarrow
\text{Generalized Eigenvalues}
}
```

---

# 1. Multivariate State Vector

Let

```math
\mathbf{Z}_t
\in
\mathbb{R}^{n}
```

represent an $n$-dimensional vector of financial variables.

For example,

```math
\mathbf{Z}_t
=
\begin{bmatrix}
B_t\\
E_t\\
D_t\\
R_t
\end{bmatrix},
```

where:

$B_t$ is log Bitcoin price,

$E_t$ is log Ethereum price,

$D_t$ is a dollar-related state,

and $R_t$ is an interest-rate state.

The variables included in a primary system must be economically motivated.

---

# 2. VAR($p$) Representation

Assume the dynamics can initially be represented by

```math
\mathbf{Z}_t
=
A_1\mathbf{Z}_{t-1}
+
A_2\mathbf{Z}_{t-2}
+
\cdots
+
A_p\mathbf{Z}_{t-p}
+
\boldsymbol{\varepsilon}_t,
```

where

```math
\boldsymbol{\varepsilon}_t
\sim
(0,\Sigma).
```

The coefficient matrices satisfy

```math
A_i
\in
\mathbb{R}^{n\times n}.
```

The error covariance matrix is

```math
\Sigma
=
E
\left[
\boldsymbol{\varepsilon}_t
\boldsymbol{\varepsilon}_t^\top
\right].
```

---

# 3. From VAR to VECM

Subtract $\mathbf{Z}_{t-1}$ from both sides.

The VAR can be rewritten as

```math
\boxed{
\Delta\mathbf{Z}_t
=
\Pi\mathbf{Z}_{t-1}
+
\sum_{i=1}^{p-1}
\Gamma_i
\Delta\mathbf{Z}_{t-i}
+
\boldsymbol{\varepsilon}_t.
}
```

The long-run matrix is

```math
\boxed{
\Pi
=
-\left(
I
-
A_1
-
A_2
-
\cdots
-
A_p
\right).
}
```

The short-run matrices are

```math
\boxed{
\Gamma_i
=
-
\sum_{j=i+1}^{p}A_j.
}
```

Thus the VECM is not a separate model class.

It is a reparameterization of the VAR designed to expose the long-run structure.

---

# 4. Interpretation of $\Pi$

The matrix

```math
\Pi
```

determines whether the variables contain unit roots and whether they are cointegrated.

There are three fundamental cases.

If

```math
\operatorname{rank}(\Pi)=0,
```

then

```math
\Pi=0.
```

There are no stationary long-run combinations represented by $\Pi$.

If

```math
\operatorname{rank}(\Pi)=n,
```

then $\Pi$ is full rank and the system is stationary in levels under the usual VECM interpretation.

The economically important case is

```math
\boxed{
0<\operatorname{rank}(\Pi)=r<n.
}
```

Then the variables are cointegrated.

---

# 5. Reduced-Rank Factorization

If

```math
\operatorname{rank}(\Pi)=r<n,
```

then $\Pi$ can be factorized as

```math
\boxed{
\Pi
=
\alpha\beta^\top,
}
```

where

```math
\alpha
\in
\mathbb{R}^{n\times r}
```

and

```math
\beta
\in
\mathbb{R}^{n\times r}.
```

The columns of $\beta$ span the cointegration space.

The columns of $\alpha$ contain the adjustment loadings.

---

# 6. Cointegrating Relations

The stationary long-run combinations are

```math
\beta^\top\mathbf{Z}_t.
```

If

```math
r=2,
```

then

```math
\beta
=
\begin{bmatrix}
| & |\\
\beta_1 & \beta_2\\
| & |
\end{bmatrix},
```

and there are two stationary relations:

```math
\beta_1^\top\mathbf{Z}_t,
```

and

```math
\beta_2^\top\mathbf{Z}_t.
```

Thus the system contains $r$ equilibrium restrictions.

---

# 7. Number of Common Stochastic Trends

For $n$ $I(1)$ variables with cointegration rank $r$,

```math
\boxed{
\text{Number of common stochastic trends}
=
n-r.
}
```

This is a fundamental dimension result.

For example, with

```math
n=4
```

and

```math
r=2,
```

four observed non-stationary variables are driven by only two independent long-run stochastic trends.

---

# 8. VECM with Cointegration Explicit

Substitute

```math
\Pi=\alpha\beta^\top
```

into the VECM:

```math
\boxed{
\Delta\mathbf{Z}_t
=
\alpha\beta^\top\mathbf{Z}_{t-1}
+
\sum_{i=1}^{p-1}
\Gamma_i
\Delta\mathbf{Z}_{t-i}
+
\boldsymbol{\varepsilon}_t.
}
```

Define the equilibrium state

```math
\mathbf{z}_{t-1}^{EC}
=
\beta^\top\mathbf{Z}_{t-1}.
```

Then

```math
\Delta\mathbf{Z}_t
=
\alpha
\mathbf{z}_{t-1}^{EC}
+
\sum_{i=1}^{p-1}
\Gamma_i
\Delta\mathbf{Z}_{t-i}
+
\boldsymbol{\varepsilon}_t.
```

This separates long-run disequilibrium from short-run dynamics.

---

# 9. Adjustment Matrix

Write

```math
\alpha
=
\begin{bmatrix}
\alpha_{11} & \cdots & \alpha_{1r}\\
\alpha_{21} & \cdots & \alpha_{2r}\\
\vdots & & \vdots\\
\alpha_{n1} & \cdots & \alpha_{nr}
\end{bmatrix}.
```

Row $i$ describes how variable $i$ responds to each equilibrium error.

For Bitcoin, if Bitcoin is the first variable, its row is

```math
\alpha_B^\top.
```

Thus its long-run adjustment component is

```math
\alpha_B^\top
\beta^\top
\mathbf{Z}_{t-1}.
```

---

# 10. Bitcoin Adjustment Equation

Let Bitcoin be the first component of $\mathbf{Z}_t$.

Then its VECM equation is

```math
\boxed{
\Delta B_t
=
\alpha_B^\top
\beta^\top
\mathbf{Z}_{t-1}
+
\sum_{i=1}^{p-1}
\gamma_{B,i}^\top
\Delta\mathbf{Z}_{t-i}
+
\varepsilon_{B,t}.
}
```

This equation contains:

```math
\text{long-run equilibrium correction}
```

and

```math
\text{short-run cross-market dynamics}.
```

Both may contain predictive information.

---

# 11. Identification Problem

The factorization

```math
\Pi=\alpha\beta^\top
```

is not unique.

Let

```math
H
```

be any invertible $r\times r$ matrix.

Then

```math
\alpha\beta^\top
=
\alpha H
H^{-1}\beta^\top.
```

Equivalently,

```math
\Pi
=
\alpha^*
{\beta^*}^\top
```

with appropriately transformed matrices.

Therefore the individual numerical representation of $\alpha$ and $\beta$ requires normalization.

The **cointegration space** is identified, but a particular basis for that space is not unique.

---

# 12. Normalization

Suppose a two-variable cointegrating relation is

```math
\beta^\top\mathbf{Z}_t
=
\beta_1Y_t
+
\beta_2X_t.
```

If

```math
\beta_1\neq0,
```

divide by $\beta_1$:

```math
Y_t
+
\frac{\beta_2}{\beta_1}
X_t.
```

A common normalization is therefore

```math
\boxed{
Y_t-\theta X_t.
}
```

Normalization changes coefficient representation.

It does not change the underlying equilibrium subspace.

---

# 13. Why Johansen Differs from Engle–Granger

Engle–Granger estimates a single equilibrium equation and then tests the estimated residual for stationarity.

Johansen instead works with the full multivariate likelihood and determines:

```math
r
=
\operatorname{rank}(\Pi).
```

It can therefore estimate multiple cointegrating vectors simultaneously.

This matters when

```math
r>1.
```

---

# 14. Partialling Out Short-Run Dynamics

The core Johansen derivation begins from

```math
\Delta\mathbf{Z}_t
=
\Pi\mathbf{Z}_{t-1}
+
\Gamma_1\Delta\mathbf{Z}_{t-1}
+
\cdots
+
\Gamma_{p-1}\Delta\mathbf{Z}_{t-p+1}
+
\varepsilon_t.
```

Define the short-run regressor vector

```math
\mathbf{W}_t
=
\begin{bmatrix}
\Delta\mathbf{Z}_{t-1}\\
\vdots\\
\Delta\mathbf{Z}_{t-p+1}\\
\text{deterministic terms}
\end{bmatrix}.
```

First regress

```math
\Delta\mathbf{Z}_t
```

on $\mathbf{W}_t$.

Let the residual be

```math
\mathbf{R}_{0t}.
```

Next regress

```math
\mathbf{Z}_{t-1}
```

on the same short-run regressors.

Let the residual be

```math
\mathbf{R}_{1t}.
```

The long-run rank problem is then studied using these residualized quantities.

---

# 15. Residualized Long-Run Regression

After partialling out short-run effects, the system becomes

```math
\boxed{
\mathbf{R}_{0t}
=
\Pi
\mathbf{R}_{1t}
+
\mathbf{u}_t.
}
```

Under cointegration,

```math
\Pi=\alpha\beta^\top,
```

so

```math
\mathbf{R}_{0t}
=
\alpha
\beta^\top
\mathbf{R}_{1t}
+
\mathbf{u}_t.
```

This is a reduced-rank multivariate regression.

The Johansen problem is essentially:

> What rank-$r$ matrix best maps lagged levels into current changes after short-run dynamics have been removed?

---

# 16. Residual Covariance Matrices

Define

```math
S_{00}
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{R}_{0t}
\mathbf{R}_{0t}^\top.
```

This is the covariance matrix of residualized changes.

Define

```math
S_{11}
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{R}_{1t}
\mathbf{R}_{1t}^\top.
```

This is the covariance matrix of residualized lagged levels.

Define the cross-covariance matrix

```math
S_{01}
=
\frac{1}{T}
\sum_{t=1}^{T}
\mathbf{R}_{0t}
\mathbf{R}_{1t}^\top.
```

Then

```math
S_{10}
=
S_{01}^\top.
```

---

# 17. Intuition Behind the Rank Problem

If lagged levels contain no long-run predictive information about current changes after short-run effects are removed, then the association between

```math
\mathbf{R}_{1t}
```

and

```math
\mathbf{R}_{0t}
```

is weak.

If certain linear combinations of lagged levels strongly explain current adjustment, those combinations correspond to candidate equilibrium relations.

Johansen identifies the strongest such stationary long-run directions.

---

# 18. Gaussian Likelihood

Assume

```math
\varepsilon_t
\sim
\mathcal{N}(0,\Sigma).
```

Conditional on regressors, the log likelihood is, up to constants,

```math
\ell
=
-\frac{T}{2}
\log|\Sigma|
-
\frac{1}{2}
\sum_{t=1}^{T}
\varepsilon_t^\top
\Sigma^{-1}
\varepsilon_t.
```

The residual under the reduced-rank model is

```math
\varepsilon_t
=
\mathbf{R}_{0t}
-
\alpha\beta^\top
\mathbf{R}_{1t}.
```

The maximum-likelihood problem therefore seeks the rank-$r$ matrices $\alpha$ and $\beta$ minimizing the residual covariance determinant.

---

# 19. Conditional Estimate of $\alpha$

For fixed $\beta$, the regression becomes

```math
\mathbf{R}_{0t}
=
\alpha
\left(
\beta^\top\mathbf{R}_{1t}
\right)
+
\mathbf{u}_t.
```

Define

```math
\mathbf{q}_t
=
\beta^\top\mathbf{R}_{1t}.
```

The least-squares / Gaussian conditional estimator of $\alpha$ is

```math
\boxed{
\hat\alpha
=
S_{01}\beta
\left(
\beta^\top
S_{11}
\beta
\right)^{-1}.
}
```

Thus once $\beta$ is known, the corresponding adjustment coefficients can be estimated directly.

---

# 20. Concentrating the Likelihood

Substituting the conditional estimate of $\alpha$ back into the likelihood removes $\alpha$ from the optimization.

The remaining problem is to choose $\beta$ such that the residual covariance is minimized subject to rank $r$.

This leads to the generalized eigenvalue problem.

---

# 21. Generalized Eigenvalue Problem

The Johansen eigenvectors solve

```math
\boxed{
\left|
\lambda S_{11}
-
S_{10}
S_{00}^{-1}
S_{01}
\right|
=
0.
}
```

Equivalently, for eigenvector $v$,

```math
S_{10}
S_{00}^{-1}
S_{01}
v
=
\lambda
S_{11}v.
```

The eigenvectors associated with the largest eigenvalues span the estimated cointegration space.

---

# 22. Whitened Interpretation

Premultiply and postmultiply using $S_{11}^{-1/2}$.

Define

```math
M
=
S_{11}^{-1/2}
S_{10}
S_{00}^{-1}
S_{01}
S_{11}^{-1/2}.
```

Then solve the ordinary eigenproblem

```math
Mw
=
\lambda w.
```

The cointegration vector is recovered through

```math
v
=
S_{11}^{-1/2}w.
```

This makes clear that Johansen is extracting directions in lagged levels that have the strongest long-run association with current adjustment after covariance normalization.

---

# 23. Eigenvalues

Order the estimated eigenvalues as

```math
1>
\hat\lambda_1
\geq
\hat\lambda_2
\geq
\cdots
\geq
\hat\lambda_n
\geq0.
```

Large eigenvalues correspond to stronger long-run relations.

If an eigenvalue is effectively zero, the associated direction contributes little evidence of cointegration.

---

# 24. Relation to Canonical Correlation

The Johansen eigenvalue problem is closely related to squared canonical correlations between:

```math
\mathbf{R}_{0t}
```

and

```math
\mathbf{R}_{1t}.
```

That is, between:

```math
\text{residualized changes}
```

and

```math
\text{residualized lagged levels}.
```

The procedure therefore finds linear combinations of levels that are most strongly related to subsequent system adjustment.

---

# 25. Maximized Likelihood Under Rank $r$

The concentrated likelihood depends on the selected eigenvalues.

Up to constants, the rank-$r$ likelihood contains terms of the form

```math
\prod_{i=1}^{r}
(1-\hat\lambda_i).
```

Additional nonzero eigenvalues improve the fit.

The rank tests determine whether that improvement is statistically sufficient to justify additional cointegrating relations.

---

# 26. Trace Statistic

To test

```math
H_0:
\operatorname{rank}(\Pi)\leq r
```

against the alternative

```math
H_1:
\operatorname{rank}(\Pi)>r,
```

the Johansen trace statistic is

```math
\boxed{
J_{\text{trace}}(r)
=
-T
\sum_{i=r+1}^{n}
\ln
\left(
1-\hat\lambda_i
\right).
}
```

The test asks whether the remaining eigenvalues beyond rank $r$ jointly contain evidence of additional cointegration.

---

# 27. Trace-Test Sequence

The test can be applied sequentially.

First test

```math
r=0.
```

If rejected, test

```math
r\leq1.
```

If rejected, test

```math
r\leq2.
```

Continue until the null is no longer rejected.

The selected rank is therefore not simply the number of visually large eigenvalues.

---

# 28. Maximum-Eigenvalue Statistic

An alternative statistic tests

```math
H_0:
\operatorname{rank}(\Pi)=r
```

against

```math
H_1:
\operatorname{rank}(\Pi)=r+1.
```

The statistic is

```math
\boxed{
J_{\max}(r,r+1)
=
-T
\ln
\left(
1-\hat\lambda_{r+1}
\right).
}
```

This tests the contribution of the next largest eigenvalue individually.

---

# 29. Why Standard Chi-Square Critical Values Do Not Apply

Under cointegration-rank null hypotheses, the asymptotic distributions of the trace and maximum-eigenvalue statistics are non-standard.

They depend on Brownian-motion functionals and on the deterministic specification.

Therefore ordinary chi-square critical values cannot simply be used.

Appropriate Johansen critical values are required.

---

# 30. Deterministic Components

Possible deterministic specifications include:

```math
\text{no intercept},
```

```math
\text{restricted intercept},
```

```math
\text{unrestricted intercept},
```

```math
\text{deterministic trend}.
```

The choice affects:

- the estimated equilibrium relation,
- the stochastic interpretation,
- the critical values.

This choice must therefore be part of the research specification rather than selected after observing favorable results.

---

# 31. Restricted Constant

Suppose the equilibrium relation contains a constant:

```math
\beta^\top\mathbf{Z}_{t-1}
+
\rho.
```

Then the VECM may be represented as

```math
\Delta\mathbf{Z}_t
=
\alpha
\left(
\beta^\top\mathbf{Z}_{t-1}
+
\rho
\right)
+
\cdots.
```

Here the constant enters the long-run equilibrium relation.

It does not necessarily imply an unrestricted deterministic drift in all differenced variables.

---

# 32. Lag-Length Selection

The Johansen procedure requires a correctly specified short-run VAR order.

If the lag order is too small, residual serial correlation may remain.

If it is excessively large:

- parameter uncertainty increases,
- effective sample size falls,
- rank inference may become unstable.

Lag selection can use:

```math
AIC,
\quad
BIC,
\quad
HQIC
```

together with residual diagnostics.

The final choice must be made using training data only.

---

# 33. Residual Diagnostics

After fitting the VECM, residuals should be examined for:

```math
\text{serial dependence},
```

```math
\text{heteroskedasticity},
```

```math
\text{heavy tails},
```

and

```math
\text{structural instability}.
```

Gaussian maximum likelihood can remain useful even when exact Gaussianity fails, but inferential assumptions and robustness must be acknowledged.

---

# 34. Bitcoin Example with One Cointegrating Relation

Suppose

```math
\mathbf{Z}_t
=
\begin{bmatrix}
B_t\\
E_t\\
D_t
\end{bmatrix}
```

and

```math
r=1.
```

Normalize the cointegrating vector on Bitcoin:

```math
\beta
=
\begin{bmatrix}
1\\
-\theta_E\\
-\theta_D
\end{bmatrix}.
```

Then the equilibrium error is

```math
\boxed{
z_t
=
B_t
-
\theta_EE_t
-
\theta_DD_t.
}
```

The adjustment vector is

```math
\alpha
=
\begin{bmatrix}
\alpha_B\\
\alpha_E\\
\alpha_D
\end{bmatrix}.
```

---

# 35. System Adjustment

The three long-run components are therefore

```math
\Delta B_t
\supset
\alpha_Bz_{t-1},
```

```math
\Delta E_t
\supset
\alpha_Ez_{t-1},
```

and

```math
\Delta D_t
\supset
\alpha_Dz_{t-1}.
```

The magnitudes determine how equilibrium restoration is distributed across markets.

---

# 36. Weak Exogeneity in the Multivariate System

For variable $j$, weak exogeneity with respect to the long-run parameters is associated with restrictions on its adjustment loading.

With one cointegrating vector, a simple case is

```math
\alpha_j=0.
```

With $r>1$, the relevant row restriction becomes

```math
\boxed{
\alpha_{j,\cdot}=0.
}
```

That variable does not respond to any of the estimated equilibrium errors.

---

# 37. Bitcoin as the Adjusting Market

Suppose

```math
\alpha_B\neq0,
```

while

```math
\alpha_E\approx0
```

and

```math
\alpha_D\approx0.
```

Then Bitcoin performs most of the long-run adjustment.

The external markets behave more like sources of the common equilibrium signal.

This configuration is potentially valuable for prediction.

---

# 38. Bitcoin as the Price Leader

Suppose instead

```math
\alpha_B\approx0,
```

while other variables have nonzero adjustment coefficients.

Then Bitcoin contributes to the common long-run trend but does not itself respond materially to equilibrium deviations.

In this case the cross-market residual may provide little direct Bitcoin return predictability.

---

# 39. Permanent Versus Transitory Components

A cointegrated system contains both:

```math
\text{permanent stochastic trends}
```

and

```math
\text{transitory equilibrium errors}.
```

With rank $r$ in an $n$-variable system:

```math
n-r
```

permanent stochastic trends remain.

The $r$ cointegrating relations correspond to transitory deviations.

This decomposition is directly relevant to price discovery.

---

# 40. Common-Trend Representation

Under suitable regularity conditions, a cointegrated system admits a common-trend representation of the form

```math
\mathbf{Z}_t
=
C
\sum_{s=1}^{t}
\boldsymbol{\varepsilon}_s
+
\mathbf{u}_t,
```

where

```math
\mathbf{u}_t
```

is stationary.

The matrix

```math
C
```

maps cumulative innovations into permanent movements in the level variables.

Cointegrating vectors satisfy approximately

```math
\beta^\top C=0.
```

Thus the cointegration space eliminates the common stochastic trends.

---

# 41. Long-Run Impact Matrix

For the VECM, the long-run impact matrix can be represented under suitable conditions using orthogonal complements of $\alpha$ and $\beta$.

Let

```math
\alpha_\perp
```

span the orthogonal complement of $\alpha$, and

```math
\beta_\perp
```

span the orthogonal complement of $\beta$.

Then a common-trend impact matrix can be written in the form

```math
\boxed{
C
=
\beta_\perp
\left[
\alpha_\perp^\top
\Gamma
\beta_\perp
\right]^{-1}
\alpha_\perp^\top,
}
```

where

```math
\Gamma
```

depends on the short-run coefficient matrices under the chosen VECM convention.

This expression identifies the directions through which innovations have permanent effects.

The exact implementation must maintain a consistent sign and coefficient convention.

---

# 42. Why Orthogonal Complements Appear

Because

```math
\beta^\top\beta_\perp=0,
```

the columns of $\beta_\perp$ span directions not eliminated by the cointegrating relations.

Those directions correspond to the permanent common-trend space.

Likewise,

```math
\alpha^\top\alpha_\perp=0.
```

The matrix $\alpha_\perp$ isolates innovations not absorbed purely through equilibrium correction.

---

# 43. Price Discovery Question

Suppose several venues or assets share the same permanent Bitcoin-related information.

The price-discovery problem is:

> Which market's innovations contribute most strongly to the permanent common stochastic trend?

This differs from asking:

> Which market moves first in a simple correlation?

Price discovery concerns the incorporation of **permanent information**, not merely contemporaneous movement.

---

# 44. Hasbrouck-Style Information Share

For cointegrated prices, one possible price-discovery framework decomposes innovations contributing to the common efficient-price process.

If the innovation covariance matrix is diagonal, the contribution of market $j$ can be related to its innovation variance weighted by its loading on the common trend.

With correlated innovations, orthogonalization is not unique.

Therefore Hasbrouck information shares typically produce ordering-dependent bounds rather than one universally identified number.

This limitation is important for fragmented cryptocurrency markets.

---

# 45. Why Variable Ordering Matters

Suppose

```math
\Sigma
```

is the innovation covariance matrix.

A Cholesky decomposition gives

```math
\Sigma
=
LL^\top.
```

If innovations are correlated, changing the variable ordering changes $L$.

Consequently the attribution of contemporaneously correlated innovations to individual markets changes.

Therefore price-discovery conclusions should not rely on a single arbitrary ordering.

---

# 46. Component-Share Interpretation

Another long-run price-discovery approach uses adjustment coefficients.

In a two-market, one-cointegration-relation system, markets that adjust less to disequilibrium can receive greater weight in the common permanent component.

The intuition is:

```math
\text{market that does not follow}
\approx
\text{market that leads}.
```

This interpretation must be applied carefully and depends on model structure.

---

# 47. Research Use of Price Discovery

For this project, cross-market price discovery may be used to identify:

```math
\text{information-leading venues},
```

```math
\text{information-leading assets},
```

and

```math
\text{markets that mainly absorb rather than originate common information}.
```

Such results could help determine which external states deserve inclusion in the predictive system.

---

# 48. Cross-Venue Bitcoin System

Suppose

```math
\mathbf{P}_t
=
\begin{bmatrix}
p_t^{Binance}\\
p_t^{Coinbase}\\
p_t^{Kraken}
\end{bmatrix}.
```

Because the instruments represent the same underlying asset, a strong economic reason exists for long-run price linkage.

If the rank is

```math
r=2,
```

then

```math
n-r
=
1.
```

There is one common permanent Bitcoin price trend.

This is a natural environment for price-discovery analysis.

---

# 49. Cross-Asset System Requires More Caution

For a system such as

```math
\begin{bmatrix}
BTC\\
ETH\\
Gold\\
DXY
\end{bmatrix},
```

the existence of cointegration is much less economically obvious.

Cointegration should therefore not be forced.

A failure to find a stable equilibrium relation is itself informative.

---

# 50. Rolling Rank Instability

Suppose the estimated cointegration rank changes through time:

```math
r_t
=
\operatorname{rank}(\Pi_t).
```

This may indicate:

- structural change,
- unstable market integration,
- sample-size effects,
- regime dependence.

A full-sample rank estimate can therefore conceal substantial temporal instability.

---

# 51. Point-in-Time Johansen Estimation

At forecast time $t$, the system must estimate:

```math
\hat r_t,
```

```math
\hat\beta_t,
```

```math
\hat\alpha_t
```

using only observations available through $t$.

Formally,

```math
\boxed{
(\hat r_t,\hat\alpha_t,\hat\beta_t)
=
f
\left(
\{\mathbf{Z}_s:s\leq t\}
\right).
}
```

Future data cannot determine the historical cointegration rank or equilibrium vector.

---

# 52. Full-Sample Rank Leakage

Suppose the full 2020–2026 sample suggests

```math
r=2.
```

Using that rank to construct features for a 2021 backtest implicitly allows future observations to determine the dimensionality of the 2021 model.

Thus even **rank selection itself** can leak future information.

This is a subtler form of leakage than merely using a future price.

---

# 53. Correct Historical Construction

For each forecast origin $t$:

```math
\text{historical sample}
\rightarrow
\text{lag selection}
\rightarrow
\text{rank test}
\rightarrow
\hat\beta_t
\rightarrow
\hat\alpha_t
\rightarrow
\text{equilibrium state}
\rightarrow
\text{forecast}.
```

Every step must be point-in-time.

---

# 54. Expanding Versus Rolling Johansen Estimation

An expanding estimator uses

```math
\{\mathbf{Z}_s:s\leq t\}.
```

A rolling estimator uses

```math
\{\mathbf{Z}_s:t-w<s\leq t\}.
```

Expanding estimation provides more observations.

Rolling estimation adapts more rapidly to structural change.

The appropriate trade-off will be tested using development data.

---

# 55. Equilibrium-State Feature Vector

Suppose rank is $r$.

Define

```math
\mathbf{e}_t
=
\hat\beta_t^\top
\mathbf{Z}_t.
```

Then

```math
\mathbf{e}_t
\in
\mathbb{R}^{r}.
```

The vector $\mathbf{e}_t$ contains the estimated point-in-time equilibrium deviations.

This becomes part of the cross-market information block.

---

# 56. Adjustment-Pressure Feature

Bitcoin's implied long-run adjustment pressure is

```math
\boxed{
A_t^{BTC}
=
\hat\alpha_{B,t}^\top
\mathbf{e}_t.
}
```

This scalar summarizes the direction and magnitude of Bitcoin's equilibrium-correction component.

It may be more directly useful for prediction than raw equilibrium residuals alone.

---

# 57. Full VECM Forecast Feature

Bitcoin's model-implied conditional adjustment is

```math
\widehat{
E_t[\Delta B_{t+1}]
}
=
\hat\alpha_{B,t}^\top
\mathbf{e}_t
+
\sum_{i=1}^{p-1}
\hat\gamma_{B,i,t}^\top
\Delta\mathbf{Z}_{t+1-i}.
```

Define

```math
\boxed{
F_t^{VECM}
=
\widehat{
E_t[\Delta B_{t+1}]
}.
}
```

This becomes a theoretically motivated cross-market feature.

---

# 58. Information-Theoretic Test

The project does not automatically trust the VECM forecast.

It tests whether

```math
\boxed{
I
\left(
Y_{t,h};
F_t^{VECM},
\mathbf{e}_t
\mid
\mathbf{p}_t
\right)
>0.
}
```

Thus even a statistically valid cointegration system must prove incremental forecasting value beyond Bitcoin's own history.

---

# 59. Domain Comparison

We can eventually compare:

```math
\mathcal{M}_{P}
```

against

```math
\mathcal{M}_{P+X},
```

where $\mathbf{X}_t$ includes:

```math
\mathbf{e}_t,
```

```math
A_t^{BTC},
```

```math
F_t^{VECM},
```

and other short-run cross-market variables.

The relevant empirical gain is

```math
\Delta_X
=
\mathcal{L}(\mathcal{M}_{P})
-
\mathcal{L}(\mathcal{M}_{P+X}).
```

---

# 60. Rank Uncertainty

The estimated rank

```math
\hat r
```

is itself uncertain.

A model that sharply changes behavior depending on whether

```math
\hat r=1
```

or

```math
\hat r=2
```

may be unstable.

The empirical work should therefore examine sensitivity to plausible nearby rank specifications where scientifically appropriate.

---

# 61. Eigenvector Instability

Even when rank is stable, estimated vectors

```math
\hat\beta_t
```

may vary substantially through time.

Possible causes include:

- genuine structural change,
- weak identification,
- collinearity,
- short estimation windows,
- changing liquidity regimes.

Therefore stability of $\beta$ is a research object, not an assumption.

---

# 62. Weak Cointegration

Suppose the relevant nonzero eigenvalue is very small.

The system may technically support cointegration while adjustment is extremely slow or economically weak.

Such a relation may be statistically detectable yet useless for short-horizon forecasting.

Statistical significance and predictive relevance must remain separate.

---

# 63. Structural Break Problem

Suppose

```math
\beta
=
\beta_1
```

before time $T_b$ and

```math
\beta
=
\beta_2
```

afterward.

A full-sample Johansen model estimates a compromise relation that may describe neither regime properly.

The project must therefore consider structural stability diagnostics before treating a long-run relation as persistent.

---

# 64. Regime-Dependent VECM

A later extension may allow

```math
\Pi_t
=
\alpha_{S_t}
\beta_{S_t}^\top.
```

Then both equilibrium structure and adjustment speed may depend on latent regime $S_t$.

This would give:

```math
\Delta\mathbf{Z}_t
=
\alpha_{S_t}
\beta_{S_t}^\top
\mathbf{Z}_{t-1}
+
\cdots.
```

This complexity will only be introduced if simpler models show evidence of instability.

---

# 65. Sparse High-Dimensional Systems

As $n$ increases, unrestricted VECMs become parameter-heavy.

A VAR($p$) contains approximately

```math
n^2p
```

lag coefficients before deterministic terms.

Therefore a large candidate asset universe cannot simply be inserted into one VECM.

Possible later solutions include:

- economically restricted systems,
- factor reduction,
- sparse VAR/VECM methods,
- shrinkage,
- hierarchical modelling.

The default strategy remains small, mechanism-driven systems.

---

# 66. Why We Avoid Data-Mining Asset Pairs

Suppose we test thousands of cross-market combinations and report only the strongest cointegration relation.

Even under no genuine structure, some apparently stable relations will emerge by chance.

This creates severe multiple-testing risk.

Therefore candidate systems should be motivated before testing.

The primary research question is information transmission, not pair mining.

---

# 67. Statistical Versus Economic Equilibrium

A stationary linear combination is a statistical equilibrium.

An economically credible equilibrium additionally requires a mechanism explaining why deviations should not persist.

Cross-venue Bitcoin prices have a clear arbitrage mechanism.

BTC and an unrelated commodity may not.

This distinction determines how strongly a discovered relation should be interpreted.

---

# 68. Falsifiable Predictions

The Johansen/VECM framework produces several testable predictions.

A mechanism-driven multivariate system may exhibit a stable rank

```math
0<r<n.
```

The estimated cointegration vectors should remain sufficiently stable through time.

Bitcoin adjustment loadings should be nonzero if equilibrium deviations directly predict Bitcoin correction.

Cross-market equilibrium features should improve out-of-sample prediction beyond the price-only benchmark.

Information-leading markets should contribute disproportionately to the permanent common trend.

These predictions may all fail.

---

# 69. What Would Reject the Multivariate Equilibrium Channel?

Evidence against this channel includes:

- rank zero,
- unstable rank,
- unstable cointegrating vectors,
- economically implausible relations,
- zero Bitcoin adjustment coefficients,
- full-sample-only results,
- no incremental out-of-sample prediction,
- apparent gains that disappear after multiple-testing correction.

Such results should be retained rather than optimized away.

---

# 70. Main Mathematical Results

The Johansen framework rests on the following core relationships.

### VECM

```math
\boxed{
\Delta\mathbf{Z}_t
=
\alpha\beta^\top\mathbf{Z}_{t-1}
+
\sum_{i=1}^{p-1}
\Gamma_i\Delta\mathbf{Z}_{t-i}
+
\boldsymbol{\varepsilon}_t
}
```

### Reduced rank

```math
\boxed{
\operatorname{rank}(\Pi)=r,
\qquad
\Pi=\alpha\beta^\top
}
```

### Generalized eigenproblem

```math
\boxed{
S_{10}
S_{00}^{-1}
S_{01}v
=
\lambda
S_{11}v
}
```

### Trace statistic

```math
\boxed{
J_{\text{trace}}(r)
=
-T
\sum_{i=r+1}^{n}
\ln(1-\hat\lambda_i)
}
```

### Maximum-eigenvalue statistic

```math
\boxed{
J_{\max}(r,r+1)
=
-T\ln(1-\hat\lambda_{r+1})
}
```

### Common stochastic trends

```math
\boxed{
n-r
}
```

### Bitcoin equilibrium pressure

```math
\boxed{
A_t^{BTC}
=
\hat\alpha_{B,t}^\top
\hat\beta_t^\top
\mathbf{Z}_t
}
```

---

# 71. Connection to the Main Research Programme

We now have a mathematically explicit cross-market chain:

```math
\boxed{
\text{Integrated Markets}
\rightarrow
\text{Cointegration Rank}
\rightarrow
\text{Equilibrium Subspace}
\rightarrow
\text{Disequilibrium State}
\rightarrow
\text{Adjustment Loadings}
\rightarrow
\text{Bitcoin Forecast}
}
```

The central empirical test remains:

```math
\boxed{
I
\left(
Y_{t,h};
\mathbf{x}_t
\mid
\mathbf{p}_t
\right)
>0,
}
```

where $\mathbf{x}_t$ contains point-in-time VECM-derived information.

The econometric machinery therefore serves the information-theoretic research question rather than becoming an isolated mathematical exercise.

---

# 72. Final Interpretation

Johansen cointegration does not tell us that Bitcoin must follow another market.

It provides a statistically disciplined way to ask:

> How many permanent stochastic trends are shared across a group of markets?

> Which linear combinations represent temporary disequilibrium?

> Which markets respond when that disequilibrium appears?

> Which innovations contribute to permanent price discovery?

And finally:

> Does any of that information improve future Bitcoin forecasts beyond Bitcoin's own historical price?

Only the last question determines whether this block survives into the final predictive architecture.
