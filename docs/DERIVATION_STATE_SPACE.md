# Derivation VI — Latent Information States and the Kalman Filter

## From Noisy Information Channels to a Dynamic Hidden Market State

The research programme assumes that observable market variables may be noisy measurements of a lower-dimensional information state.

At time $t$, we observe quantities such as:

- order-flow imbalance,
- liquidity,
- news embeddings,
- blockchain activity,
- cross-market disequilibria,
- volatility and derivatives states.

These observations need not individually represent the underlying economic state.

Instead, let

```math
\mathbf{z}_t
```

represent a latent information state that evolves through time.

The mathematical objective is to infer

```math
p
\left(
\mathbf{z}_t
\mid
\mathcal{F}_t
\right)
```

from noisy observations available up to time $t$.

The basic chain is

```math
\boxed{
\text{Latent State}
\rightarrow
\text{Noisy Observations}
\rightarrow
\text{Bayesian Updating}
\rightarrow
\text{Filtered Market State}
\rightarrow
\text{Prediction}
}
```

---

# 1. State-Space Representation

Let

```math
\mathbf{z}_t
\in
\mathbb{R}^{m}
```

denote the unobserved market state.

The transition equation is

```math
\boxed{
\mathbf{z}_t
=
A\mathbf{z}_{t-1}
+
B\mathbf{u}_t
+
\boldsymbol{\eta}_t
}
```

where:

```math
A
\in
\mathbb{R}^{m\times m}
```

governs state persistence,

```math
B\mathbf{u}_t
```

represents observable inputs,

and

```math
\boldsymbol{\eta}_t
\sim
\mathcal{N}(0,Q)
```

is state innovation noise.

---

# 2. Observation Equation

Let the observable information vector be

```math
\mathbf{y}_t
\in
\mathbb{R}^{n}.
```

The measurement equation is

```math
\boxed{
\mathbf{y}_t
=
H\mathbf{z}_t
+
\boldsymbol{\varepsilon}_t
}
```

with

```math
\boldsymbol{\varepsilon}_t
\sim
\mathcal{N}(0,R).
```

The matrix

```math
H
\in
\mathbb{R}^{n\times m}
```

maps the latent state into observed variables.

Thus observations are imperfect measurements of the hidden state.

---

# 3. Noise Assumptions

The basic Kalman model assumes

```math
E[\boldsymbol{\eta}_t]=0,
```

```math
E[\boldsymbol{\varepsilon}_t]=0,
```

and

```math
E[
\boldsymbol{\eta}_t
\boldsymbol{\varepsilon}_s^\top
]
=
0
```

for all relevant $t,s$.

State innovations satisfy

```math
E[
\boldsymbol{\eta}_t
\boldsymbol{\eta}_s^\top
]
=
\begin{cases}
Q, & t=s,\\
0, & t\neq s.
\end{cases}
```

Measurement errors satisfy

```math
E[
\boldsymbol{\varepsilon}_t
\boldsymbol{\varepsilon}_s^\top
]
=
\begin{cases}
R, & t=s,\\
0, & t\neq s.
\end{cases}
```

These assumptions can later be relaxed, but they provide the tractable baseline.

---

# 4. Prior Distribution

Suppose that after observing information through time $t-1$,

```math
\mathbf{z}_{t-1}
\mid
\mathcal{F}_{t-1}
\sim
\mathcal{N}
\left(
\hat{\mathbf{z}}_{t-1|t-1},
P_{t-1|t-1}
\right).
```

The notation

```math
\hat{\mathbf{z}}_{t-1|t-1}
```

means the estimate of state $t-1$ using observations through $t-1$.

Its uncertainty is

```math
P_{t-1|t-1}.
```

---

# 5. Prediction Step

The transition equation is

```math
\mathbf{z}_t
=
A\mathbf{z}_{t-1}
+
B\mathbf{u}_t
+
\boldsymbol{\eta}_t.
```

Take the conditional expectation given information through $t-1$:

```math
E[
\mathbf{z}_t
\mid
\mathcal{F}_{t-1}
]
=
A
E[
\mathbf{z}_{t-1}
\mid
\mathcal{F}_{t-1}
]
+
B\mathbf{u}_t.
```

Therefore

```math
\boxed{
\hat{\mathbf{z}}_{t|t-1}
=
A
\hat{\mathbf{z}}_{t-1|t-1}
+
B\mathbf{u}_t
}
```

is the prior state prediction.

---

# 6. Prediction Error

Define

```math
\tilde{\mathbf{z}}_{t|t-1}
=
\mathbf{z}_t
-
\hat{\mathbf{z}}_{t|t-1}.
```

Substituting the transition equation gives

```math
\tilde{\mathbf{z}}_{t|t-1}
=
A
\left(
\mathbf{z}_{t-1}
-
\hat{\mathbf{z}}_{t-1|t-1}
\right)
+
\boldsymbol{\eta}_t.
```

Hence

```math
\tilde{\mathbf{z}}_{t|t-1}
=
A
\tilde{\mathbf{z}}_{t-1|t-1}
+
\boldsymbol{\eta}_t.
```

---

# 7. Prediction Covariance

The prior covariance is

```math
P_{t|t-1}
=
E
\left[
\tilde{\mathbf{z}}_{t|t-1}
\tilde{\mathbf{z}}_{t|t-1}^{\top}
\right].
```

Substituting the previous expression gives

```math
P_{t|t-1}
=
A
P_{t-1|t-1}
A^\top
+
Q.
```

Thus:

```math
\boxed{
P_{t|t-1}
=
A
P_{t-1|t-1}
A^\top
+
Q
}
```

The process noise $Q$ increases uncertainty because the hidden state can change unexpectedly.

---

# 8. Predicted Observation

Before observing $\mathbf{y}_t$, the model predicts

```math
\hat{\mathbf{y}}_{t|t-1}
=
H
\hat{\mathbf{z}}_{t|t-1}.
```

The discrepancy between the actual and predicted observation is

```math
\boxed{
\boldsymbol{\nu}_t
=
\mathbf{y}_t
-
H
\hat{\mathbf{z}}_{t|t-1}
}
```

and is called the **innovation** or measurement residual.

---

# 9. Innovation Decomposition

From

```math
\mathbf{y}_t
=
H\mathbf{z}_t
+
\boldsymbol{\varepsilon}_t,
```

we obtain

```math
\boldsymbol{\nu}_t
=
H
\left(
\mathbf{z}_t
-
\hat{\mathbf{z}}_{t|t-1}
\right)
+
\boldsymbol{\varepsilon}_t.
```

Therefore

```math
\boxed{
\boldsymbol{\nu}_t
=
H
\tilde{\mathbf{z}}_{t|t-1}
+
\boldsymbol{\varepsilon}_t.
}
```

The innovation contains new information about the latent state.

---

# 10. Innovation Covariance

Define

```math
S_t
=
Var
\left(
\boldsymbol{\nu}_t
\mid
\mathcal{F}_{t-1}
\right).
```

Using independence of state and measurement errors,

```math
S_t
=
H
P_{t|t-1}
H^\top
+
R.
```

Therefore

```math
\boxed{
S_t
=
H
P_{t|t-1}
H^\top
+
R
}
```

is the innovation covariance.

---

# 11. Joint Gaussian Distribution

Before observing $\mathbf{y}_t$, the joint distribution of the latent state and observation is Gaussian:

```math
\begin{bmatrix}
\mathbf{z}_t\\
\mathbf{y}_t
\end{bmatrix}
\Bigg|
\mathcal{F}_{t-1}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
\hat{\mathbf{z}}_{t|t-1}\\
H\hat{\mathbf{z}}_{t|t-1}
\end{bmatrix},
\begin{bmatrix}
P_{t|t-1}
&
P_{t|t-1}H^\top
\\
HP_{t|t-1}
&
S_t
\end{bmatrix}
\right).
```

This joint distribution is the key to deriving the Kalman update.

---

# 12. Gaussian Conditioning Identity

Suppose

```math
\begin{bmatrix}
X\\
Y
\end{bmatrix}
\sim
\mathcal{N}
\left(
\begin{bmatrix}
\mu_X\\
\mu_Y
\end{bmatrix},
\begin{bmatrix}
\Sigma_{XX} & \Sigma_{XY}\\
\Sigma_{YX} & \Sigma_{YY}
\end{bmatrix}
\right).
```

Then

```math
X\mid Y=y
```

is Gaussian with conditional mean

```math
\boxed{
E[X\mid Y=y]
=
\mu_X
+
\Sigma_{XY}
\Sigma_{YY}^{-1}
(y-\mu_Y)
}
```

and conditional covariance

```math
\boxed{
Var(X\mid Y)
=
\Sigma_{XX}
-
\Sigma_{XY}
\Sigma_{YY}^{-1}
\Sigma_{YX}.
}
```

The Kalman filter is an application of this identity.

---

# 13. Conditional State Mean

Identify

```math
X=\mathbf{z}_t
```

and

```math
Y=\mathbf{y}_t.
```

Then

```math
\Sigma_{XY}
=
P_{t|t-1}H^\top
```

and

```math
\Sigma_{YY}
=
S_t.
```

Therefore

```math
E[
\mathbf{z}_t
\mid
\mathcal{F}_t
]
=
\hat{\mathbf{z}}_{t|t-1}
+
P_{t|t-1}
H^\top
S_t^{-1}
\boldsymbol{\nu}_t.
```

Define the Kalman gain

```math
\boxed{
K_t
=
P_{t|t-1}
H^\top
S_t^{-1}.
}
```

Hence

```math
\boxed{
\hat{\mathbf{z}}_{t|t}
=
\hat{\mathbf{z}}_{t|t-1}
+
K_t
\boldsymbol{\nu}_t.
}
```

This is the Kalman update equation.

---

# 14. Interpretation of the Kalman Gain

The gain is

```math
K_t
=
P_{t|t-1}
H^\top
\left(
HP_{t|t-1}H^\top+R
\right)^{-1}.
```

It determines how much the model changes its prior state estimate in response to new observations.

Schematically:

```math
\boxed{
\text{Posterior}
=
\text{Prior}
+
\text{Gain}
\times
\text{Surprise}.
}
```

---

# 15. Scalar Kalman Gain

Consider the scalar model

```math
z_t
=
az_{t-1}
+
\eta_t
```

and

```math
y_t
=
z_t
+
\varepsilon_t.
```

Then

```math
K_t
=
\frac{
P_{t|t-1}
}{
P_{t|t-1}+R
}.
```

Therefore:

```math
\boxed{
K_t
=
\frac{
\text{prior state uncertainty}
}{
\text{prior state uncertainty}
+
\text{measurement uncertainty}
}.
}
```

---

# 16. Limiting Cases

If measurement noise is very small,

```math
R\rightarrow0,
```

then

```math
K_t\rightarrow1
```

in the scalar direct-observation case.

The filter places high weight on the new observation.

If

```math
R\rightarrow\infty,
```

then

```math
K_t\rightarrow0.
```

The observation is extremely noisy and the filter relies primarily on its prior state estimate.

---

# 17. Posterior Covariance

Using the Gaussian conditioning formula,

```math
P_{t|t}
=
P_{t|t-1}
-
P_{t|t-1}
H^\top
S_t^{-1}
H
P_{t|t-1}.
```

Since

```math
K_t
=
P_{t|t-1}
H^\top
S_t^{-1},
```

we obtain

```math
\boxed{
P_{t|t}
=
(I-K_tH)
P_{t|t-1}.
}
```

Thus observing new information reduces state uncertainty.

---

# 18. Joseph Covariance Form

For numerical stability, the covariance update may be written as

```math
\boxed{
P_{t|t}
=
(I-K_tH)
P_{t|t-1}
(I-K_tH)^\top
+
K_tRK_t^\top.
}
```

This is algebraically equivalent under exact arithmetic but better preserves symmetry and positive semidefiniteness numerically.

---

# 19. Complete Kalman Recursion

The prediction equations are

```math
\boxed{
\hat{\mathbf{z}}_{t|t-1}
=
A\hat{\mathbf{z}}_{t-1|t-1}
+
B\mathbf{u}_t
}
```

and

```math
\boxed{
P_{t|t-1}
=
AP_{t-1|t-1}A^\top
+
Q.
}
```

The innovation is

```math
\boxed{
\boldsymbol{\nu}_t
=
\mathbf{y}_t
-
H\hat{\mathbf{z}}_{t|t-1}.
}
```

Its covariance is

```math
\boxed{
S_t
=
HP_{t|t-1}H^\top
+
R.
}
```

The gain is

```math
\boxed{
K_t
=
P_{t|t-1}H^\top S_t^{-1}.
}
```

The update equations are

```math
\boxed{
\hat{\mathbf{z}}_{t|t}
=
\hat{\mathbf{z}}_{t|t-1}
+
K_t\boldsymbol{\nu}_t
}
```

and

```math
\boxed{
P_{t|t}
=
(I-K_tH)P_{t|t-1}.
}
```

---

# 20. Bayesian Interpretation

The Kalman filter computes recursively

```math
p
\left(
\mathbf{z}_t
\mid
\mathbf{y}_{1:t}
\right).
```

The recursion contains two Bayesian operations.

### Prediction

```math
p
\left(
\mathbf{z}_t
\mid
\mathbf{y}_{1:t-1}
\right)
=
\int
p
\left(
\mathbf{z}_t
\mid
\mathbf{z}_{t-1}
\right)
p
\left(
\mathbf{z}_{t-1}
\mid
\mathbf{y}_{1:t-1}
\right)
d\mathbf{z}_{t-1}.
```

### Update

```math
p
\left(
\mathbf{z}_t
\mid
\mathbf{y}_{1:t}
\right)
\propto
p
\left(
\mathbf{y}_t
\mid
\mathbf{z}_t
\right)
p
\left(
\mathbf{z}_t
\mid
\mathbf{y}_{1:t-1}
\right).
```

The linear-Gaussian assumptions make both distributions Gaussian and therefore analytically tractable.

---

# 21. Innovation as New Information

Because

```math
\boldsymbol{\nu}_t
=
\mathbf{y}_t
-
E[
\mathbf{y}_t
\mid
\mathcal{F}_{t-1}
],
```

the innovation is the component of the current observation that was not predicted from past information.

Therefore,

```math
\boxed{
E[
\boldsymbol{\nu}_t
\mid
\mathcal{F}_{t-1}
]
=
0
}
```

under a correctly specified model.

This makes the innovation concept closely related to the project's broader focus on **new information rather than merely observed levels**.

---

# 22. Innovation Orthogonality

Under the correctly specified Kalman model,

```math
E[
\boldsymbol{\nu}_t
\boldsymbol{\nu}_s^\top
]
=
0
```

for $t\neq s$.

Persistent autocorrelation in the innovations suggests model misspecification.

Possible causes include:

- omitted dynamics,
- incorrect state dimension,
- incorrect $Q$,
- incorrect $R$,
- nonlinear relationships,
- structural breaks.

Thus innovation diagnostics become a model-validation tool.

---

# 23. Likelihood from Innovations

The predictive observation distribution is

```math
\mathbf{y}_t
\mid
\mathcal{F}_{t-1}
\sim
\mathcal{N}
\left(
H\hat{\mathbf{z}}_{t|t-1},
S_t
\right).
```

Therefore the log likelihood contribution is

```math
\ell_t
=
-\frac{1}{2}
\left[
n\log(2\pi)
+
\log|S_t|
+
\boldsymbol{\nu}_t^\top
S_t^{-1}
\boldsymbol{\nu}_t
\right].
```

Summing over time gives

```math
\boxed{
\ell
=
-\frac{1}{2}
\sum_{t=1}^{T}
\left[
n\log(2\pi)
+
\log|S_t|
+
\boldsymbol{\nu}_t^\top
S_t^{-1}
\boldsymbol{\nu}_t
\right].
}
```

This permits estimation of unknown state-space parameters by maximum likelihood.

---

# 24. Estimating $A$, $Q$ and $R$

Let the parameter vector be

```math
\theta
=
\{
A,H,Q,R,B
\}.
```

The maximum-likelihood estimator is

```math
\boxed{
\hat\theta
=
\arg\max_{\theta}
\sum_{t=1}^{T}
\ell_t(\theta).
}
```

Each candidate parameter set produces a sequence of:

- state predictions,
- innovations,
- innovation covariances,

and therefore a likelihood.

---

# 25. State Persistence

Consider the scalar state equation

```math
z_t
=
az_{t-1}
+
\eta_t.
```

If

```math
|a|<1,
```

the latent state is stationary.

If

```math
a\approx1,
```

the state is highly persistent.

If

```math
a=1,
```

the state contains a unit root.

Thus $A$ determines the persistence and mean-reversion structure of latent information.

---

# 26. Scalar State Variance

For

```math
z_t
=
az_{t-1}
+
\eta_t,
```

with

```math
Var(\eta_t)=Q,
```

a stationary unconditional variance satisfies

```math
Var(z_t)
=
a^2Var(z_{t-1})+Q.
```

Let

```math
Var(z_t)=\sigma_z^2.
```

Then

```math
\sigma_z^2
=
a^2\sigma_z^2
+
Q.
```

Therefore

```math
\boxed{
\sigma_z^2
=
\frac{Q}{1-a^2}
}
```

for $|a|<1$.

---

# 27. State Half-Life

For a deterministic deviation,

```math
E[z_{t+h}\mid z_t]
=
a^hz_t.
```

For

```math
0<a<1,
```

the half-life satisfies

```math
a^{h_{1/2}}
=
\frac{1}{2}.
```

Thus

```math
\boxed{
h_{1/2}
=
-\frac{\ln2}{\ln a}.
}
```

This gives another way to quantify persistence of a latent information state.

---

# 28. Multivariate Stability

For the vector state equation

```math
\mathbf{z}_t
=
A\mathbf{z}_{t-1}
+
\boldsymbol{\eta}_t,
```

stationarity requires all eigenvalues of $A$ to lie inside the unit circle:

```math
\boxed{
\rho(A)<1,
}
```

where $\rho(A)$ is the spectral radius.

If an eigenvalue is near one, the corresponding state direction is highly persistent.

---

# 29. Unconditional State Covariance

For stable $A$, the unconditional covariance $P$ satisfies the discrete Lyapunov equation

```math
\boxed{
P
=
APA^\top
+
Q.
}
```

Vectorizing gives

```math
\operatorname{vec}(P)
=
(A\otimes A)
\operatorname{vec}(P)
+
\operatorname{vec}(Q).
```

Therefore

```math
\left[
I-A\otimes A
\right]
\operatorname{vec}(P)
=
\operatorname{vec}(Q).
```

Hence

```math
\boxed{
\operatorname{vec}(P)
=
\left[
I-A\otimes A
\right]^{-1}
\operatorname{vec}(Q)
}
```

when the inverse exists.

---

# 30. Market Interpretation of $Q$

The process covariance

```math
Q
```

measures how rapidly the latent state itself can change unexpectedly.

Large $Q$ means:

```math
\text{state is intrinsically unstable or rapidly evolving}.
```

Small $Q$ means:

```math
\text{state evolves relatively smoothly}.
```

During information shocks, an adaptive model may require effectively larger process uncertainty.

---

# 31. Market Interpretation of $R$

The measurement covariance

```math
R
```

represents noise in the observable information channels.

Examples include:

- noisy order-book measurements,
- uncertain news sentiment,
- delayed on-chain metrics,
- noisy cross-market proxies.

A source with high measurement noise should influence the latent-state update less strongly.

---

# 32. Multimodal Observation Vector

The observation vector may be constructed as

```math
\mathbf{y}_t
=
\begin{bmatrix}
\mathbf{o}_t\\
\mathbf{n}_t\\
\mathbf{c}_t\\
\mathbf{x}_t\\
\mathbf{v}_t
\end{bmatrix}.
```

The latent state is then inferred from several noisy modalities simultaneously.

This provides a formal version of the project's multimodal information-fusion idea.

---

# 33. Domain-Specific Measurement Noise

Suppose the observation covariance is block diagonal:

```math
R
=
\begin{bmatrix}
R_O & 0 & 0 & 0\\
0 & R_N & 0 & 0\\
0 & 0 & R_C & 0\\
0 & 0 & 0 & R_X
\end{bmatrix}.
```

Then each information domain has its own measurement uncertainty.

A highly noisy news representation, for example, can have a larger covariance block

```math
R_N.
```

The filter automatically reduces its contribution to the posterior state.

---

# 34. Correlated Measurement Noise

A block-diagonal $R$ assumes conditional measurement errors are independent across domains.

That may be unrealistic.

For example:

- news and order flow may react to the same event,
- cross-market and derivatives states may share shocks.

A full covariance matrix allows

```math
R_{ij}\neq0.
```

This prevents the filter from treating correlated measurements as independent pieces of evidence.

---

# 35. Dynamic Information Weighting

Because

```math
K_t
```

depends on

```math
P_{t|t-1},
```

the optimal weight placed on new information is time dependent even if $Q$ and $R$ are constant.

Thus the system naturally performs dynamic information fusion.

This is more flexible than assigning a fixed global weight to each data source.

---

# 36. Missing Observations

Financial information sources are asynchronous.

At time $t$, some components of $\mathbf{y}_t$ may be missing.

Define a selection matrix

```math
M_t
```

that extracts only currently observed components.

Then

```math
\mathbf{y}_t^{obs}
=
M_t\mathbf{y}_t.
```

The corresponding observation matrix is

```math
H_t^{obs}
=
M_tH.
```

Measurement covariance becomes

```math
R_t^{obs}
=
M_tRM_t^\top.
```

The Kalman update can then proceed using only information actually available at time $t$.

---

# 37. Asynchronous Information Fusion

This is particularly relevant because:

- order-book data arrive continuously,
- news arrives at irregular event times,
- blockchain aggregates may update slowly,
- macro variables update according to release schedules.

The state-space system therefore offers a natural way to combine asynchronous information without pretending every source updates at the same frequency.

---

# 38. No Observation Case

If no new observation is available at time $t$, the update step can be skipped.

Then

```math
\hat{\mathbf{z}}_{t|t}
=
\hat{\mathbf{z}}_{t|t-1}
```

and

```math
P_{t|t}
=
P_{t|t-1}.
```

The model evolves using the transition dynamics until new information arrives.

---

# 39. Filtered Versus Smoothed State

The filtered estimate is

```math
\hat{\mathbf{z}}_{t|t}
=
E[
\mathbf{z}_t
\mid
\mathbf{y}_{1:t}
].
```

A smoothed estimate is

```math
\hat{\mathbf{z}}_{t|T}
=
E[
\mathbf{z}_t
\mid
\mathbf{y}_{1:T}
],
```

where

```math
T>t.
```

The smoothed state uses future observations.

Therefore:

```math
\boxed{
\hat{\mathbf{z}}_{t|T}
}
```

is inappropriate as a historical predictive feature.

---

# 40. Smoothing and Look-Ahead Bias

A smoother may provide an excellent reconstruction of historical latent states.

But if future data are used to infer the state at time $t$, then

```math
\hat{\mathbf{z}}_{t|T}
\notin
\mathcal{F}_t.
```

For predictive research, we require

```math
\boxed{
\hat{\mathbf{z}}_{t|t}
\in
\mathcal{F}_t.
}
```

This distinction is crucial.

---

# 41. Rauch–Tung–Striebel Smoother

For offline diagnostics, the RTS smoother uses

```math
J_t
=
P_{t|t}
A^\top
P_{t+1|t}^{-1}.
```

Then

```math
\hat{\mathbf{z}}_{t|T}
=
\hat{\mathbf{z}}_{t|t}
+
J_t
\left(
\hat{\mathbf{z}}_{t+1|T}
-
\hat{\mathbf{z}}_{t+1|t}
\right).
```

This is useful for latent-state interpretation and parameter estimation.

It must not be confused with a real-time trading state.

---

# 42. Latent State to Return Prediction

Suppose future Bitcoin return depends on the filtered state:

```math
R_{t,h}
=
\mathbf{c}_h^\top
\mathbf{z}_t
+
\epsilon_{t+h}.
```

Since the true state is unobserved, the real-time predictor uses

```math
\hat{\mathbf{z}}_{t|t}.
```

Thus

```math
\boxed{
\widehat{
E_t[R_{t,h}]
}
=
\mathbf{c}_h^\top
\hat{\mathbf{z}}_{t|t}.
}
```

---

# 43. Directional Probability Model

For the binary target

```math
Y_{t,h}
=
\mathbf{1}
\{
R_{t,h}>0
\},
```

a logistic state model is

```math
\boxed{
P
\left(
Y_{t,h}=1
\mid
\hat{\mathbf{z}}_{t|t}
\right)
=
\sigma
\left(
\beta_0
+
\boldsymbol{\beta}^\top
\hat{\mathbf{z}}_{t|t}
\right).
}
```

The filtered state therefore becomes a predictive representation rather than merely a descriptive latent factor.

---

# 44. State Uncertainty as a Feature

The filter produces not only

```math
\hat{\mathbf{z}}_{t|t},
```

but also

```math
P_{t|t}.
```

Therefore the model knows how uncertain it is about the latent state.

Potential predictive variables include:

```math
\operatorname{tr}(P_{t|t}),
```

```math
\det(P_{t|t}),
```

or state-specific posterior variances.

High state uncertainty may itself identify unstable market environments.

---

# 45. Information Surprise

The standardized innovation is

```math
\boxed{
\mathbf{s}_t
=
S_t^{-1/2}
\boldsymbol{\nu}_t.
}
```

Under a correctly specified Gaussian model,

```math
\mathbf{s}_t
```

should approximately have identity covariance.

Its squared Mahalanobis magnitude is

```math
\boxed{
D_t^2
=
\boldsymbol{\nu}_t^\top
S_t^{-1}
\boldsymbol{\nu}_t.
}
```

Large values represent unusually surprising information arrivals relative to the model's expectation.

---

# 46. Surprise as a Market Feature

This suggests a new research variable:

```math
\boxed{
Surprise_t
=
\boldsymbol{\nu}_t^\top
S_t^{-1}
\boldsymbol{\nu}_t.
}
```

Rather than treating all information arrivals equally, the model can measure how unexpected the observation is relative to the prior market state.

This is potentially useful for:

- news shocks,
- order-flow shocks,
- liquidity shocks,
- cross-market dislocations.

---

# 47. Decomposing Surprise by Domain

Partition the innovation vector:

```math
\boldsymbol{\nu}_t
=
\begin{bmatrix}
\boldsymbol{\nu}_t^O\\
\boldsymbol{\nu}_t^N\\
\boldsymbol{\nu}_t^C\\
\boldsymbol{\nu}_t^X
\end{bmatrix}.
```

Then domain-specific innovation magnitude can be studied separately.

This allows the project to distinguish:

```math
\text{new order-flow information}
```

from

```math
\text{new news information}
```

and

```math
\text{new cross-market information}.
```

---

# 48. State-Space Interpretation of Information Assimilation

Suppose an external information shock changes $\mathbf{y}_t$.

The innovation

```math
\boldsymbol{\nu}_t
```

updates the latent state through

```math
K_t\boldsymbol{\nu}_t.
```

Future observations may then confirm or reverse that update.

Thus the model provides a formal mechanism for:

```math
\text{Information Arrival}
\rightarrow
\text{Belief Revision}
\rightarrow
\text{Latent State Change}.
```

This is closely aligned with the core economic interpretation of the project.

---

# 49. Signal-to-Noise Interpretation

In the scalar case,

```math
K_t
=
\frac{P^-_t}
{P^-_t+R}.
```

Define the prior signal-to-noise ratio

```math
SNR_t
=
\frac{P^-_t}{R}.
```

Then

```math
K_t
=
\frac{SNR_t}
{1+SNR_t}.
```

Therefore

```math
\boxed{
K_t
=
\frac{SNR_t}
{1+SNR_t}.
}
```

This gives a direct statistical interpretation of information weighting.

---

# 50. Very Noisy Information Source

If

```math
SNR_t\ll1,
```

then approximately

```math
K_t
\approx
SNR_t.
```

The observation has little influence.

If

```math
SNR_t\gg1,
```

then

```math
K_t
\approx1.
```

The new observation dominates the prior.

---

# 51. Latent-State Compression

Suppose

```math
n\gg m.
```

Many observable features are mapped into a smaller state dimension.

Then

```math
H:
\mathbb{R}^{m}
\rightarrow
\mathbb{R}^{n}.
```

The state-space system therefore acts as a dynamic dimensionality-reduction model.

Unlike static PCA, however, the latent representation is constrained by temporal dynamics.

---

# 52. State Space Versus PCA

PCA seeks directions maximizing static variance:

```math
\max_{\|w\|=1}
Var(w^\top X_t).
```

A state-space model instead seeks a latent sequence consistent with:

```math
\text{transition dynamics}
```

and

```math
\text{measurement uncertainty}.
```

Thus a low-variance factor may still be important if it evolves predictably and influences future returns.

---

# 53. State Space Versus Autoencoder

An autoencoder learns nonlinear representation

```math
\mathbf{z}_t
=
f_\theta(\mathbf{y}_t)
```

and reconstruction

```math
\hat{\mathbf{y}}_t
=
g_\phi(\mathbf{z}_t).
```

A standard autoencoder does not automatically model:

- temporal state uncertainty,
- Bayesian updating,
- process noise,
- measurement noise.

State-space models provide these explicitly.

Later nonlinear systems may combine both ideas.

---

# 54. Parameter Identification

Latent-state models can suffer from identification issues.

For invertible matrix $T$, define transformed state

```math
\mathbf{z}_t^*
=
T\mathbf{z}_t.
```

Then a corresponding transformed model can produce the same observable distribution with

```math
A^*
=
TAT^{-1}
```

and

```math
H^*
=
HT^{-1}.
```

Thus the numerical coordinates of the latent state are not inherently economically identified.

Interpretation requires constraints or normalization.

---

# 55. Rotational Ambiguity

The previous transformation implies that multiple latent coordinate systems can represent the same observed process.

Therefore statements such as

> latent factor 1 is liquidity

require empirical justification.

Possible identification approaches include:

- structural restrictions,
- sparse loading matrices,
- domain-specific states,
- sign restrictions,
- anchored observed variables.

---

# 56. Domain-Structured Latent State

Instead of one unconstrained latent vector, define

```math
\mathbf{z}_t
=
\begin{bmatrix}
\mathbf{z}_t^O\\
\mathbf{z}_t^N\\
\mathbf{z}_t^C\\
\mathbf{z}_t^X
\end{bmatrix}.
```

The transition matrix may contain both within-domain and cross-domain effects.

For example,

```math
A
=
\begin{bmatrix}
A_{OO} & A_{ON} & A_{OC} & A_{OX}\\
A_{NO} & A_{NN} & A_{NC} & A_{NX}\\
A_{CO} & A_{CN} & A_{CC} & A_{CX}\\
A_{XO} & A_{XN} & A_{XC} & A_{XX}
\end{bmatrix}.
```

Off-diagonal blocks represent transmission between information domains.

---

# 57. Information Transmission Through $A$

Suppose

```math
A_{ON}\neq0.
```

Then the previous news state contributes to the current microstructure state.

This creates a mathematical representation of:

```math
\text{News}
\rightarrow
\text{Order Flow}.
```

Likewise,

```math
A_{XO}\neq0
```

could represent transmission from microstructure into a broader cross-market state.

Such interpretations require careful identification.

---

# 58. Exogenous Information Shocks

The control/input term

```math
B\mathbf{u}_t
```

can represent observed external events.

For example,

```math
\mathbf{u}_t
```

might contain:

- scheduled macro release surprises,
- ETF flow announcements,
- regulatory-event indicators.

Then the state transition becomes

```math
\mathbf{z}_t
=
A\mathbf{z}_{t-1}
+
B\mathbf{u}_t
+
\eta_t.
```

The matrix $B$ measures how explicit shocks alter the latent state.

---

# 59. Continuous-Time Analogue

A linear continuous-time latent state may be written

```math
d\mathbf{z}_t
=
K
\mathbf{z}_t
dt
+
Gd\mathbf{W}_t.
```

Discretizing over interval $\Delta$ gives

```math
\mathbf{z}_{t+\Delta}
=
e^{K\Delta}
\mathbf{z}_t
+
\boldsymbol{\eta}_{t+\Delta}.
```

Therefore

```math
\boxed{
A
=
e^{K\Delta}.
}
```

This provides a bridge between discrete state-space models and continuous-time stochastic processes.

---

# 60. Mean-Reversion Matrix

If the eigenvalues of $K$ have negative real parts,

```math
Re(\lambda_i(K))<0,
```

then the continuous-time state is mean reverting.

The corresponding discrete transition eigenvalues satisfy

```math
|e^{\lambda_i(K)\Delta}|<1.
```

Thus continuous and discrete stability conditions are consistent.

---

# 61. Nonlinear State Transition

Financial information transmission may be nonlinear.

A nonlinear model is

```math
\mathbf{z}_t
=
f
\left(
\mathbf{z}_{t-1},
\mathbf{u}_t
\right)
+
\boldsymbol{\eta}_t,
```

with observation equation

```math
\mathbf{y}_t
=
h(\mathbf{z}_t)
+
\boldsymbol{\varepsilon}_t.
```

The ordinary Kalman filter is no longer exact.

---

# 62. Extended Kalman Filter

The extended Kalman filter locally linearizes the nonlinear functions.

Around current state estimate,

```math
F_t
=
\left.
\frac{\partial f}
{\partial\mathbf{z}}
\right|_{\hat{\mathbf{z}}}
```

and

```math
H_t
=
\left.
\frac{\partial h}
{\partial\mathbf{z}}
\right|_{\hat{\mathbf{z}}}.
```

The standard Kalman recursions are then applied using these Jacobians.

This is only a local approximation.

---

# 63. Particle Filtering

For strongly nonlinear or non-Gaussian states, represent the posterior using particles:

```math
p
\left(
\mathbf{z}_t
\mid
\mathbf{y}_{1:t}
\right)
\approx
\sum_{i=1}^{N}
w_t^{(i)}
\delta
\left(
\mathbf{z}_t
-
\mathbf{z}_t^{(i)}
\right).
```

Particle filters permit arbitrary nonlinear state evolution but are computationally more expensive.

They are an extension, not the initial model.

---

# 64. Why We Start Linear

A linear-Gaussian model gives:

- transparent assumptions,
- exact Bayesian filtering,
- interpretable innovations,
- likelihood-based estimation,
- direct uncertainty quantification.

Only if this structure fails empirically should complexity be introduced.

This follows the project's model-complexity rule.

---

# 65. Regime Dependence

A later model may let state dynamics depend on regime $S_t$:

```math
\mathbf{z}_t
=
A_{S_t}
\mathbf{z}_{t-1}
+
\boldsymbol{\eta}_t.
```

The observation relation may also change:

```math
\mathbf{y}_t
=
H_{S_t}
\mathbf{z}_t
+
\boldsymbol{\varepsilon}_t.
```

This creates a switching state-space model.

The regime derivation is treated separately.

---

# 66. Point-in-Time Requirement

For forecasting, the latent-state feature at time $t$ must be

```math
\boxed{
\hat{\mathbf{z}}_{t|t}.
}
```

Its construction may use only:

```math
\mathbf{y}_1,\ldots,\mathbf{y}_t.
```

It may not use:

```math
\mathbf{y}_{t+1},\ldots,\mathbf{y}_{T}.
```

This is the state-space equivalent of the project's general point-in-time constraint.

---

# 67. Parameter-Learning Leakage

Even if filtering itself is causal, leakage can occur if parameters

```math
A,H,Q,R
```

are estimated using the entire historical sample.

For a forecast at time $t$, the parameter estimates should be determined from training information only:

```math
\hat\theta_t
=
f
\left(
\{\mathbf{y}_s:s<t\}
\right).
```

Then filtering proceeds using those frozen parameters over the evaluation block.

---

# 68. Walk-Forward State Estimation

For evaluation fold $k$:

```math
\mathcal{T}_k
```

is used to estimate

```math
\hat\theta_k.
```

The filter is initialized and run forward through

```math
\mathcal{E}_k
```

without re-estimating parameters using future evaluation observations.

This preserves chronological integrity.

---

# 69. Initialization

The Kalman filter requires

```math
\hat{\mathbf{z}}_{0|0}
```

and

```math
P_{0|0}.
```

Possible approaches include:

- stationary unconditional moments,
- diffuse initialization,
- training-sample estimates.

Initialization must be documented because it can influence early observations.

---

# 70. Diffuse Prior

When initial state knowledge is weak, use large uncertainty:

```math
P_{0|0}
=
\kappa I
```

with large $\kappa$.

Then early observations receive substantial influence.

The effect of initialization should diminish if the system is observable and sufficiently stable.

---

# 71. Observability

The state must be recoverable from observations.

The observability matrix is

```math
\mathcal{O}
=
\begin{bmatrix}
H\\
HA\\
HA^2\\
\vdots\\
HA^{m-1}
\end{bmatrix}.
```

The system is observable if

```math
\boxed{
\operatorname{rank}(\mathcal{O})=m.
}
```

If this fails, some latent state directions cannot be inferred from observed data.

---

# 72. Controllability of State Noise

For process-noise loading matrix $G$, the controllability matrix is

```math
\mathcal{C}
=
\begin{bmatrix}
G & AG & A^2G & \cdots & A^{m-1}G
\end{bmatrix}.
```

Full rank indicates that process shocks can excite all state directions.

These system-theoretic properties matter for stable filtering and identification.

---

# 73. Latent State Versus Raw Information

The relevant empirical comparison is not merely whether the latent state predicts returns.

We should compare:

```math
\mathcal{M}_{raw}
```

using raw information features,

against

```math
\mathcal{M}_{latent}
```

using

```math
\hat{\mathbf{z}}_{t|t}.
```

The latent representation survives only if it improves:

- predictive loss,
- stability,
- dimensional efficiency,
- interpretability.

---

# 74. Information-Theoretic Test

Let the filtered state be

```math
\hat{\mathbf{z}}_t.
```

The central question becomes

```math
\boxed{
I
\left(
Y_{t,h};
\hat{\mathbf{z}}_t
\mid
\mathbf{p}_t
\right)
>0?
}
```

A stronger comparison asks whether the latent representation preserves or enhances the useful information in the raw multimodal state.

---

# 75. Sufficiency Objective

Ideally the latent state would satisfy approximately

```math
Y_{t,h}
\perp
\mathbf{i}_t
\mid
\hat{\mathbf{z}}_t,
\mathbf{p}_t.
```

This would mean the latent state captures the predictive content of the multimodal information vector relevant to $Y$.

This is a strong objective and need not hold exactly.

---

# 76. Predictive Compression

A useful latent state should achieve

```math
\dim(\mathbf{z}_t)
\ll
\dim(\mathbf{i}_t)
```

while retaining predictive information:

```math
I
\left(
Y;
\mathbf{z}
\mid
P
\right)
\approx
I
\left(
Y;
\mathbf{i}
\mid
P
\right).
```

This expresses the desired balance between compression and predictive sufficiency.

---

# 77. State Uncertainty and Position Sizing

If later translated into trading, posterior uncertainty can influence action.

Suppose probability edge is high but

```math
\operatorname{tr}(P_{t|t})
```

is also high.

The system may reduce exposure when the estimated latent state is poorly identified.

This creates a potential connection between Bayesian state uncertainty and risk management.

---

# 78. Falsifiable Predictions

The latent-state framework generates several empirical predictions.

### Prediction 1

Filtered latent states should contain incremental predictive information beyond price history.

### Prediction 2

The latent representation should outperform or simplify raw multimodal concatenation.

### Prediction 3

Large standardized innovations should identify unusually informative market events.

### Prediction 4

Domain-specific measurement noise should alter the effective information weights.

### Prediction 5

State persistence should vary across information dimensions or regimes.

### Prediction 6

Using smoothed states should appear stronger historically than filtered states; only the filtered result is legitimate for forecasting.

---

# 79. What Would Reject the Latent-State Layer?

The latent-state architecture should be dropped or simplified if:

- filtering adds no out-of-sample value,
- states are unstable or poorly identified,
- innovations remain strongly autocorrelated,
- raw features perform better,
- parameter estimates are excessively unstable,
- the results depend on smoothed rather than filtered states.

The project does not require a latent model merely because the title contains the word latent.

---

# 80. Mathematical Core

The main state-space equations are:

### Transition

```math
\boxed{
\mathbf{z}_t
=
A\mathbf{z}_{t-1}
+
B\mathbf{u}_t
+
\boldsymbol{\eta}_t
}
```

### Observation

```math
\boxed{
\mathbf{y}_t
=
H\mathbf{z}_t
+
\boldsymbol{\varepsilon}_t
}
```

### Prediction

```math
\boxed{
\hat{\mathbf{z}}_{t|t-1}
=
A\hat{\mathbf{z}}_{t-1|t-1}
+
B\mathbf{u}_t
}
```

### Prior covariance

```math
\boxed{
P_{t|t-1}
=
AP_{t-1|t-1}A^\top
+
Q
}
```

### Innovation

```math
\boxed{
\boldsymbol{\nu}_t
=
\mathbf{y}_t
-
H\hat{\mathbf{z}}_{t|t-1}
}
```

### Innovation covariance

```math
\boxed{
S_t
=
HP_{t|t-1}H^\top
+
R
}
```

### Kalman gain

```math
\boxed{
K_t
=
P_{t|t-1}
H^\top
S_t^{-1}
}
```

### Posterior state

```math
\boxed{
\hat{\mathbf{z}}_{t|t}
=
\hat{\mathbf{z}}_{t|t-1}
+
K_t\boldsymbol{\nu}_t
}
```

### Posterior covariance

```math
\boxed{
P_{t|t}
=
(I-K_tH)P_{t|t-1}
}
```

---

# 81. Connection to the Main Research Thesis

The latent-state component can now be expressed as:

```math
\boxed{
\text{Multimodal Observations}
\rightarrow
\text{Expected Information}
\rightarrow
\text{Innovation}
\rightarrow
\text{Bayesian State Update}
\rightarrow
\text{Filtered Information State}
\rightarrow
\text{Bitcoin Forecast}
}
```

The project is therefore not simply proposing a generic hidden factor.

It proposes a real-time probabilistic representation of the information state inferred from noisy and asynchronous financial observations.

The central empirical test remains:

```math
\boxed{
I
\left(
Y_{t,h};
\hat{\mathbf{z}}_{t|t}
\mid
\mathbf{p}_t
\right)
>0.
}
```

Only if that relationship survives point-in-time out-of-sample testing does the latent-state layer remain in the final architecture.
