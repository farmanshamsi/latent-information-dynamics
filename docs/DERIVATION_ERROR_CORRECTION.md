# Derivation IV — Error Correction and Cross-Market Adjustment

## From Cointegration to Predictive Price Dynamics

Cointegration establishes that several non-stationary variables may share a stable long-run equilibrium relation.

It does **not** by itself establish which variable adjusts when that equilibrium is disturbed.

This document derives the error-correction representation that connects:

```math
\text{Long-Run Equilibrium}
\rightarrow
\text{Disequilibrium}
\rightarrow
\text{Short-Run Adjustment}
\rightarrow
\text{Potential Predictive Information}.
```

For the Bitcoin research programme, the central question is whether cross-market disequilibrium observed at time $t$ helps predict subsequent Bitcoin adjustment.

---

# 1. Starting Point: A Bivariate VAR

Let

```math
\mathbf{Z}_t
=
\begin{bmatrix}
Y_t\\
X_t
\end{bmatrix}.
```

Consider a first-order vector autoregression:

```math
\mathbf{Z}_t
=
A\mathbf{Z}_{t-1}
+
\boldsymbol{\varepsilon}_t.
```

Writing the coefficient matrix explicitly,

```math
A
=
\begin{bmatrix}
a_{11} & a_{12}\\
a_{21} & a_{22}
\end{bmatrix},
```

gives

```math
Y_t
=
a_{11}Y_{t-1}
+
a_{12}X_{t-1}
+
\varepsilon_{Y,t},
```

and

```math
X_t
=
a_{21}Y_{t-1}
+
a_{22}X_{t-1}
+
\varepsilon_{X,t}.
```

If $Y_t$ and $X_t$ are individually $I(1)$, modelling their levels using an unrestricted stationary VAR interpretation is problematic.

The model can instead be rewritten in differences.

---

# 2. Reparameterization

Subtract $\mathbf{Z}_{t-1}$ from both sides:

```math
\mathbf{Z}_t-\mathbf{Z}_{t-1}
=
A\mathbf{Z}_{t-1}
-
\mathbf{Z}_{t-1}
+
\boldsymbol{\varepsilon}_t.
```

Therefore,

```math
\Delta\mathbf{Z}_t
=
(A-I)\mathbf{Z}_{t-1}
+
\boldsymbol{\varepsilon}_t.
```

Define

```math
\Pi
=
A-I.
```

Then

```math
\boxed{
\Delta\mathbf{Z}_t
=
\Pi\mathbf{Z}_{t-1}
+
\boldsymbol{\varepsilon}_t
}
```

is the error-correction representation of the VAR(1).

The matrix $\Pi$ contains the long-run structure.

---

# 3. Why the Rank of $\Pi$ Matters

Consider three cases.

### Case 1: $\operatorname{rank}(\Pi)=0$

Then

```math
\Pi=0,
```

so

```math
\Delta\mathbf{Z}_t
=
\boldsymbol{\varepsilon}_t.
```

There is no long-run equilibrium restriction.

The variables behave as independent stochastic trends after accounting for short-run dynamics.

---

### Case 2: $\operatorname{rank}(\Pi)=n$

Then $\Pi$ has full rank.

The level process is stationary rather than genuinely $I(1)$.

---

### Case 3: $0<\operatorname{rank}(\Pi)=r<n$

This is the cointegrated case.

The reduced-rank matrix can be factorized as

```math
\boxed{
\Pi
=
\alpha\beta^\top
}
```

where:

```math
\beta
\in
\mathbb{R}^{n\times r}
```

contains the cointegrating vectors, and

```math
\alpha
\in
\mathbb{R}^{n\times r}
```

contains the adjustment coefficients.

This decomposition is fundamental.

---

# 4. Economic Meaning of $\beta$

For a single cointegrating relation,

```math
\beta^\top\mathbf{Z}_{t-1}
```

is stationary.

For two variables we may normalize

```math
\beta
=
\begin{bmatrix}
1\\
-\theta
\end{bmatrix}.
```

Then

```math
\beta^\top\mathbf{Z}_{t-1}
=
Y_{t-1}
-
\theta X_{t-1}.
```

Including an intercept gives the disequilibrium term

```math
\boxed{
z_{t-1}
=
Y_{t-1}
-
c
-
\theta X_{t-1}.
}
```

Thus $\beta$ defines **what equilibrium means**.

---

# 5. Economic Meaning of $\alpha$

The vector

```math
\alpha
=
\begin{bmatrix}
\alpha_Y\\
\alpha_X
\end{bmatrix}
```

determines how each variable reacts when

```math
z_{t-1}\neq0.
```

Substituting

```math
\Pi=\alpha\beta^\top
```

into the VEC representation gives

```math
\Delta\mathbf{Z}_t
=
\alpha
\beta^\top
\mathbf{Z}_{t-1}
+
\boldsymbol{\varepsilon}_t.
```

Therefore

```math
\Delta Y_t
=
\alpha_Y z_{t-1}
+
\varepsilon_{Y,t},
```

and

```math
\Delta X_t
=
\alpha_X z_{t-1}
+
\varepsilon_{X,t}.
```

Hence:

```math
\boxed{
\beta
=
\text{equilibrium structure}
}
```

while

```math
\boxed{
\alpha
=
\text{equilibrium-adjustment mechanism}.
}
```

---

# 6. Sign of the Adjustment Coefficient

Suppose

```math
z_{t-1}
=
Y_{t-1}
-
c
-
\theta X_{t-1}.
```

If

```math
z_{t-1}>0,
```

then $Y$ is above its equilibrium relation with $X$.

For $Y$ to restore equilibrium by falling,

```math
\Delta Y_t<0.
```

Since

```math
\Delta Y_t
=
\alpha_Yz_{t-1}
+
\varepsilon_{Y,t},
```

equilibrium restoration through $Y$ requires approximately

```math
\boxed{
\alpha_Y<0.
}
```

Similarly, $X$ may contribute to restoration depending on the normalization and sign of $\theta$.

The sign cannot be interpreted independently of the chosen cointegrating-vector normalization.

---

# 7. Speed of Adjustment

Suppose only $Y$ adjusts and $X$ is temporarily fixed.

Then

```math
\Delta Y_t
=
\alpha_Yz_{t-1}.
```

Because

```math
z_t
=
Y_t-c-\theta X_t,
```

and $X_t=X_{t-1}$,

```math
z_t-z_{t-1}
=
\Delta Y_t.
```

Therefore

```math
z_t
=
z_{t-1}
+
\alpha_Yz_{t-1}.
```

Hence

```math
\boxed{
z_t
=
(1+\alpha_Y)z_{t-1}.
}
```

Define

```math
\phi
=
1+\alpha_Y.
```

Then

```math
z_t
=
\phi z_{t-1}.
```

Mean reversion requires

```math
|\phi|<1.
```

Therefore,

```math
\boxed{
-2<\alpha_Y<0.
}
```

under this highly simplified one-adjusting-variable representation.

---

# 8. Monotonic Versus Oscillatory Adjustment

If

```math
0<\phi<1,
```

equilibrium deviations decay monotonically.

Since

```math
\phi=1+\alpha_Y,
```

this corresponds to

```math
-1<\alpha_Y<0.
```

If instead

```math
-1<\phi<0,
```

the error changes sign during adjustment.

This corresponds to

```math
-2<\alpha_Y<-1.
```

The system remains stable but convergence oscillates around equilibrium.

---

# 9. Equilibrium Half-Life from the ECM

From

```math
z_{t+h}
=
\phi^h z_t,
```

define the half-life by

```math
|\phi|^{h_{1/2}}
=
\frac{1}{2}.
```

For positive $\phi$,

```math
h_{1/2}
=
-\frac{\ln2}{\ln\phi}.
```

Because

```math
\phi=1+\alpha_Y,
```

we obtain

```math
\boxed{
h_{1/2}
=
-\frac{\ln2}
{\ln(1+\alpha_Y)}
}
```

when $-1<\alpha_Y<0$.

This connects the ECM adjustment coefficient directly to the speed of equilibrium restoration.

---

# 10. General Distributed-Lag Relationship

Now consider a richer dynamic system:

```math
Y_t
=
c
+
\sum_{i=1}^{p}a_iY_{t-i}
+
\sum_{j=0}^{q}b_jX_{t-j}
+
\varepsilon_t.
```

Such a model contains both:

- short-run effects,
- long-run equilibrium information.

An ECM reorganizes these into separate components.

---

# 11. ARDL(1,1) Derivation

Consider

```math
Y_t
=
c
+
\phi Y_{t-1}
+
\beta_0X_t
+
\beta_1X_{t-1}
+
\varepsilon_t.
```

Subtract $Y_{t-1}$:

```math
\Delta Y_t
=
c
+
(\phi-1)Y_{t-1}
+
\beta_0X_t
+
\beta_1X_{t-1}
+
\varepsilon_t.
```

Use

```math
X_t
=
X_{t-1}
+
\Delta X_t.
```

Then

```math
\Delta Y_t
=
c
+
(\phi-1)Y_{t-1}
+
\beta_0\Delta X_t
+
(\beta_0+\beta_1)X_{t-1}
+
\varepsilon_t.
```

---

# 12. Extracting the Long-Run Equilibrium

Factor $(\phi-1)$ from the level terms:

```math
\Delta Y_t
=
\beta_0\Delta X_t
+
(\phi-1)
\left[
Y_{t-1}
-
\frac{c}{1-\phi}
-
\frac{\beta_0+\beta_1}{1-\phi}
X_{t-1}
\right]
+
\varepsilon_t.
```

Define

```math
\alpha
=
\phi-1,
```

```math
\mu
=
\frac{c}{1-\phi},
```

and

```math
\theta
=
\frac{\beta_0+\beta_1}{1-\phi}.
```

Then

```math
\boxed{
\Delta Y_t
=
\beta_0\Delta X_t
+
\alpha
\left(
Y_{t-1}
-
\mu
-
\theta X_{t-1}
\right)
+
\varepsilon_t.
}
```

This is an error-correction model.

---

# 13. Separation of Time Scales

The term

```math
\beta_0\Delta X_t
```

captures the short-run effect of changes in $X$.

The term

```math
Y_{t-1}
-
\mu
-
\theta X_{t-1}
```

captures deviation from long-run equilibrium.

The coefficient

```math
\alpha
```

controls the rate of correction.

Therefore the ECM explicitly separates:

```math
\boxed{
\text{Short-Run Dynamics}
+
\text{Long-Run Disequilibrium Correction}.
}
```

---

# 14. Why Differencing Alone Is Incomplete

A model such as

```math
\Delta Y_t
=
\gamma\Delta X_t+\varepsilon_t
```

removes stochastic trends.

But if $X_t$ and $Y_t$ are cointegrated, differencing alone discards the stationary quantity

```math
z_{t-1}
=
Y_{t-1}
-
\mu
-
\theta X_{t-1}.
```

The ECM retains this information:

```math
\Delta Y_t
=
\gamma\Delta X_t
+
\alpha z_{t-1}
+
\varepsilon_t.
```

This is why cointegration matters for prediction.

---

# 15. Engle–Granger Representation Principle

The fundamental representation result can be stated informally as follows.

If a set of $I(1)$ variables is cointegrated, then their dynamics admit an error-correction representation.

Conversely, a stable error-correction representation implies a long-run equilibrium restriction.

Schematically:

```math
\boxed{
\text{Cointegration}
\Longleftrightarrow
\text{Error Correction}
}
```

under the appropriate regularity conditions.

This provides the bridge between long-run equilibrium and short-run dynamics.

---

# 16. Higher-Order VECM

For a VAR($p$),

```math
\mathbf{Z}_t
=
A_1\mathbf{Z}_{t-1}
+
\cdots
+
A_p\mathbf{Z}_{t-p}
+
\boldsymbol{\varepsilon}_t,
```

the VECM representation is

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

where

```math
\Pi
=
-\left(
I-\sum_{i=1}^{p}A_i
\right),
```

and

```math
\Gamma_i
=
-
\sum_{j=i+1}^{p}A_j.
```

If

```math
\operatorname{rank}(\Pi)=r,
```

then

```math
\Pi=\alpha\beta^\top.
```

---

# 17. Deriving the VAR(2) Case Explicitly

Start with

```math
\mathbf{Z}_t
=
A_1\mathbf{Z}_{t-1}
+
A_2\mathbf{Z}_{t-2}
+
\boldsymbol{\varepsilon}_t.
```

Subtract $\mathbf{Z}_{t-1}$:

```math
\Delta\mathbf{Z}_t
=
(A_1-I)\mathbf{Z}_{t-1}
+
A_2\mathbf{Z}_{t-2}
+
\boldsymbol{\varepsilon}_t.
```

Use

```math
\mathbf{Z}_{t-2}
=
\mathbf{Z}_{t-1}
-
\Delta\mathbf{Z}_{t-1}.
```

Then

```math
\Delta\mathbf{Z}_t
=
(A_1+A_2-I)\mathbf{Z}_{t-1}
-
A_2\Delta\mathbf{Z}_{t-1}
+
\boldsymbol{\varepsilon}_t.
```

Define

```math
\Pi=A_1+A_2-I
```

and

```math
\Gamma_1=-A_2.
```

Therefore

```math
\boxed{
\Delta\mathbf{Z}_t
=
\Pi\mathbf{Z}_{t-1}
+
\Gamma_1\Delta\mathbf{Z}_{t-1}
+
\boldsymbol{\varepsilon}_t.
}
```

The VECM is therefore a reparameterization of the VAR, not an unrelated model.

---

# 18. Bitcoin Two-Asset ECM

Let

```math
B_t=\log BTC_t
```

and let

```math
X_t
```

represent a candidate cross-market price.

Suppose the equilibrium relation is

```math
z_t
=
B_t
-
c
-
\theta X_t.
```

Then a Bitcoin adjustment equation is

```math
\boxed{
\Delta B_t
=
\alpha_Bz_{t-1}
+
\sum_{i=1}^{p}\phi_i\Delta B_{t-i}
+
\sum_{j=0}^{q}\gamma_j\Delta X_{t-j}
+
\varepsilon_{B,t}.
}
```

This equation contains two conceptually different sources of prediction.

---

# 19. Short-Run Predictive Information

The terms

```math
\Delta X_{t-j}
```

capture recent changes in the external market.

If

```math
\gamma_j\neq0,
```

lagged movements in $X$ may contain predictive information for Bitcoin.

This corresponds to a lead-lag mechanism.

---

# 20. Long-Run Predictive Information

The term

```math
z_{t-1}
```

captures accumulated deviation from equilibrium.

If

```math
\alpha_B\neq0,
```

Bitcoin participates in correcting that deviation.

Thus:

```math
\boxed{
\alpha_B\neq0
}
```

is essential if the equilibrium residual is to represent a direct Bitcoin-adjustment mechanism.

---

# 21. Example of Adjustment

Suppose

```math
z_{t-1}>0.
```

Bitcoin is relatively expensive compared with the equilibrium relation.

If

```math
\alpha_B<0,
```

then

```math
\alpha_Bz_{t-1}<0.
```

Hence

```math
E[\Delta B_t\mid z_{t-1}]<0
```

all else equal.

This gives a directional prediction.

---

# 22. No Bitcoin Adjustment

Suppose instead

```math
\alpha_B=0.
```

Then

```math
z_{t-1}
```

does not directly affect Bitcoin's change equation.

Cointegration may still exist because another market performs the adjustment.

For example,

```math
\Delta X_t
=
\alpha_Xz_{t-1}
+
\cdots
```

with

```math
\alpha_X\neq0.
```

This illustrates why cointegration alone does not imply Bitcoin predictability.

---

# 23. Weak Exogeneity

Within the cointegration system, variable $X$ is weakly exogenous for the long-run parameter $\beta$ when its adjustment coefficient is zero:

```math
\boxed{
\alpha_X=0.
}
```

In that case, $X$ does not respond to deviations from the long-run relation.

Instead, the other variable must perform the equilibrium correction.

For Bitcoin research, a particularly interesting configuration is:

```math
\alpha_X\approx0
```

and

```math
\alpha_B\neq0.
```

This suggests that the external market helps define the common equilibrium while Bitcoin performs the adjustment.

---

# 24. Price Leadership Interpretation

Suppose

```math
\alpha_B<0
```

and

```math
\alpha_X\approx0.
```

Then deviations from equilibrium are corrected mainly through Bitcoin.

This is consistent with $X$ behaving as a long-run price leader relative to Bitcoin.

Conversely, if

```math
\alpha_B\approx0
```

and

```math
\alpha_X\neq0,
```

Bitcoin may be the price-leading variable.

These are statistical adjustment interpretations, not automatically structural causal claims.

---

# 25. Both Markets Can Adjust

There is no requirement that only one market corrects disequilibrium.

We may observe

```math
\alpha_B\neq0
```

and

```math
\alpha_X\neq0.
```

Then both variables participate in restoring equilibrium.

The relative magnitudes and signs describe the adjustment allocation.

---

# 26. Error-Correction Forecast

At prediction time $t$, suppose all required variables are observable.

The one-step conditional Bitcoin forecast is

```math
E_t[\Delta B_{t+1}]
=
\alpha_Bz_t
+
\sum_i\phi_i\Delta B_{t+1-i}
+
\sum_j\gamma_j\Delta X_{t+1-j}.
```

Therefore the equilibrium error is a point-in-time predictive feature.

---

# 27. Forecast Decomposition

Define:

```math
F_t^{EC}
=
\alpha_Bz_t
```

and

```math
F_t^{SR}
=
\sum_i\phi_i\Delta B_{t+1-i}
+
\sum_j\gamma_j\Delta X_{t+1-j}.
```

Then

```math
\boxed{
E_t[\Delta B_{t+1}]
=
F_t^{EC}
+
F_t^{SR}.
}
```

This allows us to measure separately:

- long-run disequilibrium prediction,
- short-run cross-market prediction.

---

# 28. Multi-Horizon Adjustment

If the simplified disequilibrium process follows

```math
z_{t+1}
=
\phi z_t+\eta_{t+1},
```

then

```math
E_t[z_{t+h}]
=
\phi^hz_t.
```

The expected amount of disequilibrium removed by horizon $h$ is

```math
z_t-E_t[z_{t+h}]
=
(1-\phi^h)z_t.
```

Therefore:

```math
\boxed{
\text{Expected equilibrium correction by }h
=
(1-\phi^h)z_t.
}
```

This gives a natural link between the adjustment speed and forecast horizon.

---

# 29. Continuous-Time Approximation

For small sampling interval $\Delta$, suppose

```math
\phi=e^{-\kappa\Delta}.
```

Then

```math
\kappa
=
-\frac{\ln\phi}{\Delta}.
```

The continuous-time equilibrium error satisfies approximately

```math
dz_t
=
-\kappa z_tdt
+
\sigma dW_t.
```

Its half-life is

```math
\boxed{
h_{1/2}
=
\frac{\ln2}{\kappa}.
}
```

Thus discrete ECM adjustment and continuous OU mean reversion are closely related.

---

# 30. Error-Correction Versus Statistical Arbitrage

A stationary equilibrium error does not automatically create a profitable trading strategy.

Profitability also depends on:

```math
\text{Adjustment Speed},
```

```math
\text{Transaction Costs},
```

```math
\text{Spread},
```

```math
\text{Liquidity},
```

```math
\text{Parameter Stability}.
```

The present research uses the ECM primarily as an **information representation**, not as an automatic pairs-trading rule.

---

# 31. Time-Varying Equilibrium

Suppose the long-run coefficient changes over time:

```math
z_t
=
B_t-c_t-\theta_tX_t.
```

Then a fixed full-sample estimate

```math
\hat\theta
```

may create artificial disequilibrium.

Possible later approaches include:

- rolling estimation,
- expanding estimation,
- recursive least squares,
- state-space coefficients,
- regime-dependent cointegration.

These extensions are justified only if the fixed relation is empirically unstable.

---

# 32. Estimation Error in the Disequilibrium Feature

The true equilibrium error is

```math
z_t
=
B_t-c-\theta X_t.
```

But empirically we observe

```math
\hat z_t
=
B_t-\hat c-\hat\theta X_t.
```

Hence

```math
\hat z_t-z_t
=
-(\hat c-c)
-
(\hat\theta-\theta)X_t.
```

The predictive feature therefore contains parameter-estimation error.

This error can be particularly important in short rolling windows.

---

# 33. Point-in-Time Estimation

At time $t$, the estimated equilibrium parameters must satisfy

```math
(\hat c_t,\hat\theta_t)
=
f
\left(
\{B_s,X_s:s\leq t\}
\right).
```

They may not use observations with

```math
s>t.
```

Thus the feature used in a historical backtest must be

```math
\boxed{
\hat z_t^{PIT}
=
B_t
-
\hat c_t
-
\hat\theta_tX_t.
}
```

A full-sample residual retroactively applied to earlier dates creates look-ahead bias.

---

# 34. Why Full-Sample Cointegration Can Leak

Suppose $\hat\theta$ is estimated using data from 2020–2026.

Using that coefficient to construct a 2021 predictive feature allows the 2021 model indirectly to learn from 2022–2026 observations.

Thus even though

```math
B_{2021}
```

and

```math
X_{2021}
```

are historical, the transformation itself contains future information.

This is transformation leakage.

---

# 35. Correct Walk-Forward Construction

For each prediction date $t$:

1. use observations available through $t$,
2. estimate the cointegrating relation,
3. test or validate the relation using training data,
4. calculate $\hat z_t$,
5. generate the forecast,
6. advance the historical clock.

Mathematically,

```math
\hat\beta_t
=
\arg\min_{\beta}
\mathcal{L}
\left(
\{Z_s:s\leq t\}
\right).
```

Then

```math
\hat z_t
=
\hat\beta_t^\top Z_t.
```

---

# 36. Short-Run Granger Predictability

Within the Bitcoin ECM,

```math
\Delta B_t
=
\alpha_Bz_{t-1}
+
\sum_i\phi_i\Delta B_{t-i}
+
\sum_j\gamma_j\Delta X_{t-j}
+
\varepsilon_t,
```

short-run predictive information from $X$ can be tested through

```math
H_0:
\gamma_1=\gamma_2=\cdots=\gamma_q=0.
```

Rejecting this restriction indicates that lagged changes in $X$ help predict Bitcoin changes within the specified system.

---

# 37. Long-Run Predictive Channel

The long-run adjustment hypothesis is

```math
H_0:
\alpha_B=0.
```

Against

```math
H_1:
\alpha_B\neq0.
```

If $\alpha_B$ is non-zero, Bitcoin responds systematically to equilibrium deviations.

Thus short-run and long-run predictability are separate hypotheses.

---

# 38. Joint Predictive Hypothesis

A broader cross-market hypothesis is:

```math
H_0:
\alpha_B=0
\quad\text{and}\quad
\gamma_1=\cdots=\gamma_q=0.
```

If this joint restriction holds, the candidate market contributes neither:

- disequilibrium correction,
- lagged short-run prediction

to the Bitcoin equation.

---

# 39. ECM and Conditional Information

Define the cross-market feature block:

```math
\mathbf{x}_t
=
\begin{bmatrix}
\hat z_t\\
\Delta X_t\\
\Delta X_{t-1}\\
\vdots
\end{bmatrix}.
```

The project's information-theoretic question becomes

```math
\boxed{
I
\left(
Y_{t,h};
\mathbf{x}_t
\mid
\mathbf{p}_t
\right)
>0?
}
```

Thus the ECM does not replace the main framework.

It produces theoretically motivated cross-market information variables for that framework.

---

# 40. Directional Probability Representation

For binary target

```math
Y_{t,h}
=
\mathbf{1}
\{
R_{t,h}^{BTC}>0
\},
```

a probability model may use

```math
P
\left(
Y_{t,h}=1
\mid
\hat z_t,
\Delta X_t,
\mathbf{p}_t
\right)
=
\sigma
\left(
\beta_0
+
\beta_z\hat z_t
+
\boldsymbol{\beta}_X^\top\Delta\mathbf{X}_t
+
\boldsymbol{\beta}_P^\top\mathbf{p}_t
\right).
```

The empirical question is not whether $\hat z_t$ merely appears significant in sample.

It is whether it reduces genuine out-of-sample probabilistic loss.

---

# 41. Nonlinear Adjustment

Linear error correction assumes

```math
\frac{\partial\Delta B_t}
{\partial z_{t-1}}
=
\alpha_B.
```

But real markets may exhibit transaction-cost bands.

A threshold ECM could instead use

```math
\Delta B_t
=
\begin{cases}
\alpha_-z_{t-1}
+
\cdots,
&
z_{t-1}<-c,
\\
\text{short-run dynamics},
&
|z_{t-1}|\leq c,
\\
\alpha_+z_{t-1}
+
\cdots,
&
z_{t-1}>c.
\end{cases}
```

This allows adjustment to activate only when disequilibrium is sufficiently large.

It is an optional extension rather than the default model.

---

# 42. Regime-Dependent Adjustment

Let $S_t$ denote a latent market regime.

Then

```math
\Delta B_t
=
\alpha_{B,S_t}z_{t-1}
+
\Gamma_{S_t}^\top W_t
+
\varepsilon_t.
```

Thus the speed of equilibrium restoration may depend on:

- volatility,
- liquidity,
- risk appetite,
- market stress.

This later connects ECM dynamics to the project's regime-switching architecture.

---

# 43. Cross-Market Price Discovery

Suppose Bitcoin and another market are cointegrated.

The adjustment coefficients

```math
\alpha_B
```

and

```math
\alpha_X
```

help identify where long-run equilibrium information is incorporated.

If one market barely reacts to disequilibrium while the other adjusts strongly, the first behaves more like the source of the common permanent price signal.

This motivates later formal price-discovery measures.

---

# 44. Falsifiable Bitcoin Predictions

The ECM framework generates concrete predictions.

### Prediction 1

If Bitcoin corrects equilibrium deviations,

```math
\alpha_B\neq0.
```

### Prediction 2

If another market leads Bitcoin in the long run,

```math
\alpha_X\approx0
```

while

```math
\alpha_B\neq0.
```

### Prediction 3

If short-run changes in another market lead Bitcoin,

```math
\exists j:
\gamma_j\neq0.
```

### Prediction 4

A point-in-time disequilibrium feature should improve out-of-sample forecasts if cross-market equilibrium contains incremental information.

### Prediction 5

The predictive contribution should disappear if the relationship breaks structurally.

All predictions are falsifiable.

---

# 45. What Would Reject the ECM Channel?

The cross-market ECM mechanism would be weakened if:

- no stable cointegration is found,
- $\alpha_B$ is effectively zero,
- the relation is structurally unstable,
- point-in-time residuals do not predict Bitcoin,
- predictive performance disappears out of sample,
- improvements exist only after full-sample parameter estimation.

Thus statistical cointegration alone is insufficient evidence.

---

# 46. Mathematical Core

The main relationships derived in this document are:

### VAR to VECM

```math
\boxed{
\Delta\mathbf{Z}_t
=
\Pi\mathbf{Z}_{t-1}
+
\sum_{i=1}^{p-1}
\Gamma_i\Delta\mathbf{Z}_{t-i}
+
\varepsilon_t
}
```

### Reduced-rank decomposition

```math
\boxed{
\Pi
=
\alpha\beta^\top
}
```

### Equilibrium error

```math
\boxed{
z_t
=
\beta^\top\mathbf{Z}_t
}
```

### Bitcoin ECM

```math
\boxed{
\Delta B_t
=
\alpha_Bz_{t-1}
+
\sum_i\phi_i\Delta B_{t-i}
+
\sum_j\gamma_j\Delta X_{t-j}
+
\varepsilon_{B,t}
}
```

### Simplified equilibrium dynamics

```math
\boxed{
z_t
=
(1+\alpha_B)z_{t-1}
}
```

### Adjustment half-life

```math
\boxed{
h_{1/2}
=
-\frac{\ln2}
{\ln(1+\alpha_B)}
}
```

under the simplified monotonic one-adjusting-variable system.

---

# 47. Connection to the Research Thesis

The cross-market component can now be expressed as:

```math
\boxed{
\text{Common Stochastic Trend}
\rightarrow
\text{Equilibrium Relation}
\rightarrow
\text{Disequilibrium}
\rightarrow
\text{Adjustment Pressure}
\rightarrow
\text{Future Bitcoin Return}
}
```

The central empirical question is:

```math
\boxed{
I
\left(
Y_{t,h};
z_t,\Delta X_t,\Delta X_{t-1},\ldots
\mid
\mathbf{p}_t
\right)
>0?
}
```

This is substantially stronger than testing whether Bitcoin is correlated with another asset.

It asks whether a theoretically derived, point-in-time cross-market state contains predictive information beyond Bitcoin's own historical price.

---

# 48. Next Mathematical Step

The two-variable ECM is useful for intuition and hypothesis construction.

A multimarket system requires simultaneous estimation of:

```math
\text{cointegration rank},
```

```math
\text{cointegrating vectors},
```

and

```math
\text{adjustment coefficients}.
```

The next derivation therefore develops the Johansen framework:

```math
\boxed{
\text{VAR}
\rightarrow
\text{VECM}
\rightarrow
\Pi
\rightarrow
\operatorname{rank}(\Pi)
\rightarrow
\alpha\beta^\top
\rightarrow
\text{generalized eigenvalue problem}.
}
```

That provides the mathematical foundation for multivariate Bitcoin price-discovery systems.
