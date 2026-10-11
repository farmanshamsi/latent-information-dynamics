# Derivation VII — Regime Switching and State-Dependent Information Transmission

## Hidden Market Regimes, Bayesian Filtering and Conditional Predictive Structure

The research programme does not assume that information is incorporated into Bitcoin price at a constant rate.

A news shock may matter differently during:

- high versus low liquidity,
- stressed versus normal markets,
- high versus low volatility,
- strongly trending versus range-bound markets.

The same order-flow imbalance may therefore carry different predictive content depending on the state of the market.

This motivates a latent regime variable

```math
S_t
\in
\{1,\ldots,K\}.
```

The central mathematical chain is

```math
\boxed{
\text{Hidden Regime}
\rightarrow
\text{Transition Probability}
\rightarrow
\text{Prior Regime Belief}
\rightarrow
\text{Observation Likelihood}
\rightarrow
\text{Posterior Regime Belief}
\rightarrow
\text{Regime-Conditional Forecast}
}
```

---

# 1. Hidden Regime Variable

Let

```math
S_t
\in
\{1,\ldots,K\}
```

denote the unobserved regime at time $t$.

For example, in a two-state model:

```math
S_t
=
\begin{cases}
1, & \text{low-volatility / high-liquidity regime},\\
2, & \text{high-volatility / low-liquidity regime}.
\end{cases}
```

These interpretations are not imposed automatically.

The latent states must be characterized empirically after estimation.

---

# 2. First-Order Markov Property

Assume the regime process is first-order Markov:

```math
P
\left(
S_t=j
\mid
S_{t-1},S_{t-2},\ldots
\right)
=
P
\left(
S_t=j
\mid
S_{t-1}
\right).
```

Thus the probability of the next state depends only on the current state.

Define

```math
p_{ij}
=
P
\left(
S_t=j
\mid
S_{t-1}=i
\right).
```

---

# 3. Transition Matrix

Collect transition probabilities into

```math
P
=
\begin{bmatrix}
p_{11} & p_{12} & \cdots & p_{1K}\\
p_{21} & p_{22} & \cdots & p_{2K}\\
\vdots & \vdots & \ddots & \vdots\\
p_{K1} & p_{K2} & \cdots & p_{KK}
\end{bmatrix}.
```

Each row satisfies

```math
\boxed{
\sum_{j=1}^{K}p_{ij}=1.
}
```

and

```math
0\leq p_{ij}\leq1.
```

---

# 4. Two-State Transition Matrix

For $K=2$,

```math
P
=
\begin{bmatrix}
p_{11} & 1-p_{11}\\
1-p_{22} & p_{22}
\end{bmatrix}.
```

Here:

```math
p_{11}
=
P(S_t=1\mid S_{t-1}=1)
```

and

```math
p_{22}
=
P(S_t=2\mid S_{t-1}=2)
```

measure regime persistence.

---

# 5. Regime Probability Vector

Let

```math
\boldsymbol{\pi}_t
=
\begin{bmatrix}
P(S_t=1)\\
\vdots\\
P(S_t=K)
\end{bmatrix}.
```

If the convention uses column probability vectors, then propagation is

```math
\boxed{
\boldsymbol{\pi}_t
=
P^\top
\boldsymbol{\pi}_{t-1}.
}
```

This equation describes how regime probabilities evolve before observing new data.

---

# 6. Stationary Regime Distribution

A stationary regime distribution satisfies

```math
\boldsymbol{\pi}
=
P^\top
\boldsymbol{\pi}.
```

Thus

```math
\boxed{
(P^\top-I)
\boldsymbol{\pi}
=
0
}
```

subject to

```math
\boxed{
\mathbf{1}^\top
\boldsymbol{\pi}
=
1.
}
```

The stationary distribution is therefore an eigenvector of $P^\top$ associated with eigenvalue one.

---

# 7. Two-State Stationary Distribution

For

```math
P
=
\begin{bmatrix}
p_{11} & 1-p_{11}\\
1-p_{22} & p_{22}
\end{bmatrix},
```

let the stationary probabilities be

```math
\pi_1
```

and

```math
\pi_2=1-\pi_1.
```

Stationarity requires

```math
\pi_1
=
p_{11}\pi_1
+
(1-p_{22})\pi_2.
```

Substitute

```math
\pi_2=1-\pi_1.
```

Then

```math
\pi_1
=
p_{11}\pi_1
+
(1-p_{22})(1-\pi_1).
```

Expanding,

```math
\pi_1
=
p_{11}\pi_1
+
1-p_{22}
-
(1-p_{22})\pi_1.
```

Collecting $\pi_1$ terms gives

```math
\pi_1
\left(
2-p_{11}-p_{22}
\right)
=
1-p_{22}.
```

Hence

```math
\boxed{
\pi_1
=
\frac{
1-p_{22}
}{
2-p_{11}-p_{22}
}.
}
```

Similarly,

```math
\boxed{
\pi_2
=
\frac{
1-p_{11}
}{
2-p_{11}-p_{22}
}.
}
```

---

# 8. Expected Regime Duration

Suppose the system is currently in regime $i$.

The probability of remaining in that regime for exactly $d$ periods before leaving is geometric:

```math
P(D_i=d)
=
p_{ii}^{d-1}
(1-p_{ii}),
```

for

```math
d=1,2,\ldots
```

The expected duration is therefore

```math
E[D_i]
=
\sum_{d=1}^{\infty}
d
p_{ii}^{d-1}
(1-p_{ii}).
```

Using

```math
\sum_{d=1}^{\infty}
dx^{d-1}
=
\frac{1}{(1-x)^2},
```

we obtain

```math
E[D_i]
=
(1-p_{ii})
\frac{1}{(1-p_{ii})^2}.
```

Therefore

```math
\boxed{
E[D_i]
=
\frac{1}{1-p_{ii}}.
}
```

---

# 9. Interpretation of Persistence

If

```math
p_{ii}=0.50,
```

then

```math
E[D_i]=2.
```

If

```math
p_{ii}=0.95,
```

then

```math
E[D_i]=20.
```

Thus transition probabilities have a direct economic interpretation in terms of expected regime duration.

---

# 10. Regime-Conditional Observation Distribution

Suppose observed variable $Y_t$ has regime-specific distribution

```math
Y_t
\mid
S_t=j
\sim
f_j
\left(
y_t;
\theta_j
\right).
```

For a Gaussian switching model,

```math
Y_t
\mid
S_t=j
\sim
\mathcal{N}
\left(
\mu_j,
\sigma_j^2
\right).
```

Thus

```math
f_j(y_t)
=
\frac{
1
}{
\sqrt{2\pi\sigma_j^2}
}
\exp
\left[
-\frac{
(y_t-\mu_j)^2
}{
2\sigma_j^2
}
\right].
```

---

# 11. Regime-Switching Return Model

A basic model is

```math
R_t
=
\mu_{S_t}
+
\sigma_{S_t}
\varepsilon_t,
```

where

```math
\varepsilon_t
\sim
\mathcal{N}(0,1).
```

Thus both expected return and volatility may depend on the hidden regime.

A more conservative specification may allow only volatility to switch:

```math
R_t
=
\mu
+
\sigma_{S_t}\varepsilon_t.
```

---

# 12. Regime-Conditional Information Model

For this research, the regime may affect the predictive value of the information state.

Define

```math
P
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,
S_t=j
\right)
=
f_j(\mathbf{i}_t).
```

For logistic specialists,

```math
\boxed{
P
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,
S_t=j
\right)
=
\sigma
\left(
\beta_{0,j}
+
\boldsymbol{\beta}_j^\top
\mathbf{i}_t
\right).
}
```

Thus the predictive coefficients may change across regimes.

---

# 13. Why Regimes Matter

Suppose order-flow imbalance is

```math
OFI_t.
```

A regime-dependent return model could be

```math
E
\left[
R_{t,h}
\mid
OFI_t,S_t=j
\right]
=
\beta_j OFI_t.
```

Then

```math
\beta_1
\neq
\beta_2
```

means identical order flow has different expected price impact in different regimes.

This is economically plausible if liquidity differs by state.

---

# 14. Prior Regime Probability

Let

```math
\xi_{t-1|t-1}^{(i)}
=
P
\left(
S_{t-1}=i
\mid
\mathcal{F}_{t-1}
\right).
```

Before observing data at time $t$, predict the probability of regime $j$:

```math
\xi_{t|t-1}^{(j)}
=
\sum_{i=1}^{K}
p_{ij}
\xi_{t-1|t-1}^{(i)}.
```

Therefore

```math
\boxed{
\xi_{t|t-1}^{(j)}
=
P
\left(
S_t=j
\mid
\mathcal{F}_{t-1}
\right).
}
```

---

# 15. Matrix Prediction Form

Let

```math
\boldsymbol{\xi}_{t-1|t-1}
=
\begin{bmatrix}
\xi_{t-1|t-1}^{(1)}\\
\vdots\\
\xi_{t-1|t-1}^{(K)}
\end{bmatrix}.
```

Then

```math
\boxed{
\boldsymbol{\xi}_{t|t-1}
=
P^\top
\boldsymbol{\xi}_{t-1|t-1}.
}
```

This is the regime-prediction step.

---

# 16. Observation Likelihood by Regime

After observing $y_t$, calculate

```math
f_j(y_t)
=
p
\left(
y_t
\mid
S_t=j,
\mathcal{F}_{t-1}
\right).
```

Each regime assigns a likelihood to the new observation.

A high-volatility regime may assign much greater probability to an extreme return than a low-volatility regime.

---

# 17. Total Predictive Density

Before knowing the current regime, the predictive density is a mixture:

```math
p
\left(
y_t
\mid
\mathcal{F}_{t-1}
\right)
=
\sum_{j=1}^{K}
f_j(y_t)
\xi_{t|t-1}^{(j)}.
```

Therefore

```math
\boxed{
f(y_t\mid\mathcal{F}_{t-1})
=
\sum_{j=1}^{K}
\xi_{t|t-1}^{(j)}
f_j(y_t).
}
```

This is a finite mixture distribution.

---

# 18. Hamilton Filter Update

Bayes' rule gives

```math
P
\left(
S_t=j
\mid
\mathcal{F}_t
\right)
=
\frac{
p(y_t\mid S_t=j,\mathcal{F}_{t-1})
P(S_t=j\mid\mathcal{F}_{t-1})
}{
p(y_t\mid\mathcal{F}_{t-1})
}.
```

Substituting notation gives

```math
\boxed{
\xi_{t|t}^{(j)}
=
\frac{
f_j(y_t)
\xi_{t|t-1}^{(j)}
}{
\sum_{k=1}^{K}
f_k(y_t)
\xi_{t|t-1}^{(k)}
}.
}
```

This is the central Hamilton-filter update.

---

# 19. Interpretation of the Hamilton Update

The posterior regime probability is proportional to

```math
\boxed{
\text{prior regime probability}
\times
\text{likelihood of the new observation}.
}
```

Thus unexpected market observations shift probability toward regimes capable of explaining them.

This is Bayesian belief updating over hidden market states.

---

# 20. Two-State Example

Suppose prior probabilities are

```math
\xi_{t|t-1}^{(1)}=0.80
```

and

```math
\xi_{t|t-1}^{(2)}=0.20.
```

Suppose an extreme observation has likelihoods

```math
f_1(y_t)=0.01
```

and

```math
f_2(y_t)=0.20.
```

The unnormalized posterior masses are

```math
0.80\times0.01=0.008
```

and

```math
0.20\times0.20=0.040.
```

Therefore the observation moves substantial posterior probability toward regime 2 despite its lower prior probability.

---

# 21. Log Likelihood

The one-step predictive density is

```math
f(y_t\mid\mathcal{F}_{t-1})
=
\sum_{j=1}^{K}
\xi_{t|t-1}^{(j)}
f_j(y_t).
```

The sample log likelihood is

```math
\boxed{
\ell(\theta)
=
\sum_{t=1}^{T}
\log
\left[
\sum_{j=1}^{K}
\xi_{t|t-1}^{(j)}
f_j(y_t;\theta_j)
\right].
}
```

The parameters may include:

```math
P,
```

```math
\mu_j,
```

```math
\sigma_j,
```

and other regime-specific coefficients.

---

# 22. Maximum-Likelihood Estimation

Let

```math
\theta
```

contain all unknown parameters.

Then

```math
\boxed{
\hat\theta
=
\arg\max_{\theta}
\ell(\theta).
}
```

Each candidate parameter vector implies an entire recursive sequence of regime probabilities.

Therefore the likelihood must be evaluated through the filtering recursion.

---

# 23. Transition Probability Constraints

Directly optimizing

```math
p_{ij}
```

requires enforcing

```math
0\leq p_{ij}\leq1
```

and

```math
\sum_jp_{ij}=1.
```

A convenient parameterization uses softmax probabilities:

```math
p_{ij}
=
\frac{
e^{\gamma_{ij}}
}{
\sum_{k=1}^{K}
e^{\gamma_{ik}}
}.
```

This automatically ensures valid transition probabilities.

---

# 24. Two-State Logistic Parameterization

For a two-state model, persistence may be written

```math
p_{11}
=
\frac{
e^{\gamma_1}
}{
1+e^{\gamma_1}
}
```

and

```math
p_{22}
=
\frac{
e^{\gamma_2}
}{
1+e^{\gamma_2}
}.
```

Optimization can proceed over unconstrained real-valued parameters $\gamma_1,\gamma_2$.

---

# 25. Filtered Regime Probability

The real-time state probability is

```math
\boxed{
P
\left(
S_t=j
\mid
\mathcal{F}_t
\right).
}
```

This uses only observations available through time $t$.

It is therefore suitable as a point-in-time predictive feature.

---

# 26. Smoothed Regime Probability

After observing the full sample through $T>t$, one may calculate

```math
P
\left(
S_t=j
\mid
\mathcal{F}_T
\right).
```

This is a smoothed regime probability.

It uses future information.

Therefore:

```math
\boxed{
P(S_t=j\mid\mathcal{F}_T)
}
```

must not be used as a historical trading or predictive feature.

---

# 27. Smoothing Recursion

Define

```math
\xi_{t|T}^{(i)}
=
P
\left(
S_t=i
\mid
\mathcal{F}_T
\right).
```

A backward smoothing recursion can be written

```math
\boxed{
\xi_{t|T}^{(i)}
=
\xi_{t|t}^{(i)}
\sum_{j=1}^{K}
\frac{
p_{ij}
\xi_{t+1|T}^{(j)}
}{
\xi_{t+1|t}^{(j)}
}.
}
```

This is useful for historical interpretation.

It is not available in real time.

---

# 28. Filtering Versus Smoothing Leakage

A common mistake is to identify historical regimes using smoothed probabilities and then use those labels as if they had been observable at the time.

That creates look-ahead bias.

The correct forecasting feature is

```math
\boxed{
\boldsymbol{\xi}_{t|t}.
}
```

not

```math
\boldsymbol{\xi}_{t|T}.
```

---

# 29. Hard Versus Soft Regime Classification

A hard regime assignment is

```math
\hat S_t
=
\arg\max_j
\xi_{t|t}^{(j)}.
```

But this discards uncertainty.

A soft regime representation retains the full probability vector

```math
\boldsymbol{\xi}_{t|t}.
```

For prediction, the soft representation is generally more informative.

---

# 30. Regime Entropy

Regime uncertainty can be measured by entropy:

```math
\boxed{
H_t^S
=
-
\sum_{j=1}^{K}
\xi_{t|t}^{(j)}
\log
\xi_{t|t}^{(j)}.
}
```

Low entropy means the model is confident about the current regime.

High entropy means the state is ambiguous.

---

# 31. Maximum Regime Entropy

Entropy is maximized when all regimes are equally probable:

```math
\xi_{t|t}^{(j)}
=
\frac{1}{K}.
```

Then

```math
H_{\max}^S
=
\log K.
```

A normalized uncertainty measure is

```math
\boxed{
\widetilde H_t^S
=
\frac{
H_t^S
}{
\log K
}.
}
```

Hence

```math
0
\leq
\widetilde H_t^S
\leq
1.
```

---

# 32. Regime-Weighted Forecast

Suppose each regime produces probability

```math
p_{t,h}^{(j)}
=
P
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,S_t=j
\right).
```

Since the true regime is unobserved, the overall forecast is

```math
\boxed{
p_{t,h}
=
\sum_{j=1}^{K}
\xi_{t|t}^{(j)}
p_{t,h}^{(j)}.
}
```

This is a mixture-of-experts probability forecast.

---

# 33. Derivation by Total Probability

By the law of total probability,

```math
P(Y=1\mid\mathcal{F}_t)
=
\sum_j
P(Y=1,S_t=j\mid\mathcal{F}_t).
```

Factor the joint probability:

```math
P(Y=1,S_t=j\mid\mathcal{F}_t)
=
P(Y=1\mid S_t=j,\mathcal{F}_t)
P(S_t=j\mid\mathcal{F}_t).
```

Therefore

```math
\boxed{
P(Y=1\mid\mathcal{F}_t)
=
\sum_j
p_t^{(j)}
\xi_{t|t}^{(j)}.
}
```

---

# 34. Regime-Dependent Information Coefficients

Suppose information vector $\mathbf{i}_t$ enters a logistic model.

Then

```math
\operatorname{logit}
\left(
p_t^{(j)}
\right)
=
\beta_{0,j}
+
\boldsymbol{\beta}_j^\top
\mathbf{i}_t.
```

The coefficient difference

```math
\boldsymbol{\beta}_j
-
\boldsymbol{\beta}_k
```

measures how information transmission changes between regimes $j$ and $k$.

---

# 35. Constant Versus Regime-Switching Model

The null model is

```math
\boldsymbol{\beta}_1
=
\boldsymbol{\beta}_2
=
\cdots
=
\boldsymbol{\beta}_K.
```

Under this restriction, regime conditioning adds no predictive slope variation.

The alternative is that at least two regimes satisfy

```math
\boldsymbol{\beta}_j
\neq
\boldsymbol{\beta}_k.
```

This is the formal state-dependence hypothesis.

---

# 36. Regime-Dependent Microstructure Impact

From the microstructure derivation,

```math
\Delta m_t
\approx
\frac{\kappa}{D_t}
OFI_t.
```

Allow the impact coefficient to switch:

```math
\boxed{
\Delta m_t
=
\lambda_{S_t}
OFI_t
+
\varepsilon_t.
}
```

If

```math
\lambda_2>\lambda_1,
```

the same order-flow shock moves price more strongly in regime 2.

This may occur if regime 2 has lower effective depth.

---

# 37. Regime-Dependent Information Decay

Suppose information value decays according to

```math
G_j(h)
=
G_{0,j}
e^{-\lambda_jh}
```

within regime $j$.

Then the information half-life is

```math
\boxed{
h_{1/2}^{(j)}
=
\frac{\ln2}{\lambda_j}.
}
```

Hence information may be incorporated rapidly in liquid regimes and more slowly in stressed regimes.

---

# 38. Regime-Dependent ECM

From the cross-market model,

```math
\Delta B_t
=
\alpha_Bz_{t-1}
+
\cdots
```

Allow equilibrium adjustment to depend on regime:

```math
\boxed{
\Delta B_t
=
\alpha_{B,S_t}
z_{t-1}
+
\Gamma_{S_t}^\top
\mathbf{w}_t
+
\varepsilon_t.
}
```

Then equilibrium restoration speed changes across market states.

---

# 39. Regime-Dependent Equilibrium Half-Life

Under the simplified relation

```math
z_t
=
(1+\alpha_{B,j})z_{t-1}
```

in regime $j$, the half-life is

```math
\boxed{
h_{1/2}^{(j)}
=
-\frac{
\ln2
}{
\ln(1+\alpha_{B,j})
}.
}
```

Thus the model can quantify whether cross-market disequilibria persist longer in some regimes than others.

---

# 40. Switching Latent-State Model

Combine the previous Kalman framework with regimes:

```math
\mathbf{z}_t
=
A_{S_t}
\mathbf{z}_{t-1}
+
\boldsymbol{\eta}_t.
```

The observation equation may be

```math
\mathbf{y}_t
=
H_{S_t}
\mathbf{z}_t
+
\boldsymbol{\varepsilon}_t.
```

Then both latent-state evolution and information measurement depend on regime.

This is a switching state-space model.

---

# 41. Why Switching Kalman Filtering Is Harder

In a standard Kalman filter, the posterior remains Gaussian.

With hidden discrete regimes, the state distribution becomes a mixture:

```math
p
\left(
\mathbf{z}_t
\mid
\mathcal{F}_t
\right)
=
\sum_{j=1}^{K}
P(S_t=j\mid\mathcal{F}_t)
p
\left(
\mathbf{z}_t
\mid
S_t=j,\mathcal{F}_t
\right).
```

Over time, the number of possible regime histories grows exponentially.

Approximation or mixture-collapsing methods may therefore be required.

This is an advanced extension rather than the baseline.

---

# 42. Regime-Specific Multimodal Weights

Suppose specialist probabilities are

```math
p_t^O,
\quad
p_t^N,
\quad
p_t^C,
\quad
p_t^X.
```

A regime-specific fusion model may be

```math
\operatorname{logit}
\left(
p_t^{(j)}
\right)
=
\beta_{0,j}
+
\beta_{O,j}
\operatorname{logit}(p_t^O)
+
\beta_{N,j}
\operatorname{logit}(p_t^N)
```

```math
+
\beta_{C,j}
\operatorname{logit}(p_t^C)
+
\beta_{X,j}
\operatorname{logit}(p_t^X).
```

Thus the relevance of each information source can vary across regimes.

---

# 43. Example Interpretation

A possible empirical pattern might be:

```math
\beta_{O,\text{stress}}
>
\beta_{O,\text{normal}}
```

while

```math
\beta_{N,\text{stress}}
<
\beta_{N,\text{normal}}.
```

This would suggest microstructure dominates during stressed trading while machine-readable news contributes more under normal conditions.

Such interpretations must come from estimated results rather than assumptions.

---

# 44. Time-Varying Transition Probabilities

The baseline Markov model assumes constant transition probabilities:

```math
p_{ij}
=
\text{constant}.
```

A later extension may allow

```math
p_{ij,t}
=
P
\left(
S_t=j
\mid
S_{t-1}=i,
\mathbf{x}_{t-1}
\right).
```

Thus external state variables may affect the probability of regime transition.

---

# 45. Logistic Transition Model

For a two-state system, one could define

```math
P
\left(
S_t=1
\mid
S_{t-1}=1,
\mathbf{x}_{t-1}
\right)
=
\sigma
\left(
\gamma_0
+
\boldsymbol{\gamma}^\top
\mathbf{x}_{t-1}
\right).
```

This creates time-varying transition probabilities.

It should only be introduced if constant-transition models are empirically inadequate.

---

# 46. Regime Probability as a Feature

Even without regime-specific predictive models, filtered probabilities themselves may be useful:

```math
\mathbf{r}_t
=
\boldsymbol{\xi}_{t|t}.
```

Then compare

```math
\mathcal{M}_{P}
```

with

```math
\mathcal{M}_{P+R}.
```

The question is whether estimated regime state contains incremental predictive information.

---

# 47. Information-Theoretic Regime Hypothesis

The direct regime hypothesis is

```math
\boxed{
I
\left(
Y_{t,h};
S_t
\mid
\mathbf{p}_t
\right)
>0.
}
```

Since $S_t$ is latent, empirically we use filtered regime probabilities:

```math
\boxed{
I
\left(
Y_{t,h};
\boldsymbol{\xi}_{t|t}
\mid
\mathbf{p}_t
\right)
>0?
}
```

---

# 48. Conditional Information by Regime

The value of information block $I$ may differ across regimes.

Define

```math
G_j
=
I
\left(
Y;
I
\mid
P,
S=j
\right).
```

Then the state-dependent-information hypothesis is

```math
\boxed{
G_j
\neq
G_k
}
```

for at least some regimes $j\neq k$.

This directly formalizes the idea of regime-dependent information assimilation.

---

# 49. Regime-Specific Domain Contribution

For domain $d$, define

```math
G_{d,j}
=
I
\left(
Y;
I_d
\mid
P,I_{-d},S=j
\right).
```

This allows questions such as:

> Does order flow contain more unique predictive information during high-volatility regimes?

> Does news become more useful during low-microstructure-noise regimes?

This is more informative than calculating one global feature importance.

---

# 50. Regime Classification Is Not the Main Goal

The project does not primarily care whether a state should be called:

> bull,

> bear,

> panic,

or

> calm.

The important question is whether conditioning on the latent state improves the model of **information transmission into price**.

Thus economic interpretation follows predictive and statistical evidence.

---

# 51. Label Switching

Hidden-state models possess a label-identification problem.

If regimes 1 and 2 are exchanged while their parameters are also exchanged, the likelihood is unchanged.

Therefore:

```math
\boxed{
\text{Regime labels are arbitrary without identifying restrictions.}
}
```

A state becomes “high volatility” only because its estimated variance is larger—not because the number 2 intrinsically means high volatility.

---

# 52. Identification by Ordering

A simple identification convention might impose

```math
\sigma_1^2
<
\sigma_2^2.
```

Then state 1 is defined as the lower-volatility state.

Alternatively one may order by:

- liquidity,
- volatility,
- mean return,
- another economically meaningful parameter.

The convention must be documented.

---

# 53. Too Many Regimes

Increasing $K$ always increases flexibility.

But excessive regimes can produce:

- unstable estimates,
- nearly empty states,
- poor out-of-sample performance,
- difficult interpretation.

Therefore the project will begin with

```math
K=2
```

and increase complexity only when justified.

---

# 54. Model Selection

Candidate regime counts may be compared using:

```math
AIC,
\quad
BIC,
```

together with:

- out-of-sample likelihood,
- regime persistence,
- interpretability,
- parameter stability.

A better in-sample likelihood alone is insufficient.

---

# 55. Degenerate Regimes

A warning sign occurs when one regime has:

```math
P(S_t=j)
\approx0
```

for almost the entire sample.

Another warning sign is a regime with extremely small variance used merely to fit a handful of observations.

Such solutions may be statistical degeneracies rather than meaningful market states.

---

# 56. Structural Break Versus Recurring Regime

A regime-switching model assumes states can recur.

A structural break instead represents a more permanent parameter change.

These are conceptually different.

If a state appears only once and never returns, a structural-break model may be more appropriate than a Markov regime model.

---

# 57. Persistence Versus Predictability

A highly persistent regime can be easy to classify because

```math
p_{ii}
```

is large.

But that does not imply the regime improves return prediction.

Regime identification and predictive usefulness must be evaluated separately.

---

# 58. Forecasting the Next Regime

Given filtered probabilities at time $t$,

```math
\boldsymbol{\xi}_{t|t},
```

the one-step regime forecast is

```math
\boxed{
\boldsymbol{\xi}_{t+1|t}
=
P^\top
\boldsymbol{\xi}_{t|t}.
}
```

For horizon $h$,

```math
\boxed{
\boldsymbol{\xi}_{t+h|t}
=
(P^\top)^h
\boldsymbol{\xi}_{t|t}.
}
```

Thus transition dynamics provide explicit multi-horizon regime forecasts.

---

# 59. Long-Horizon Regime Distribution

If the Markov chain is irreducible and aperiodic, then

```math
(P^\top)^h
\boldsymbol{\xi}_{t|t}
```

converges toward the stationary distribution:

```math
\boxed{
\lim_{h\rightarrow\infty}
(P^\top)^h
\boldsymbol{\xi}_{t|t}
=
\boldsymbol{\pi}.
}
```

Therefore current regime information gradually loses influence at long horizons.

---

# 60. Regime Probability Shock

Suppose an observation sharply increases the probability of stress regime 2.

Define

```math
\Delta\xi_t^{(2)}
=
\xi_{t|t}^{(2)}
-
\xi_{t|t-1}^{(2)}.
```

This quantity measures how much the new observation changed belief about the current market state.

It may itself be interpreted as a state-surprise measure.

---

# 61. Bayesian Regime Surprise

The information content of the posterior update can be measured through KL divergence:

```math
\boxed{
D_{KL}
\left(
\boldsymbol{\xi}_{t|t}
\Vert
\boldsymbol{\xi}_{t|t-1}
\right)
=
\sum_{j=1}^{K}
\xi_{t|t}^{(j)}
\log
\frac{
\xi_{t|t}^{(j)}
}{
\xi_{t|t-1}^{(j)}
}.
}
```

Large values indicate that new observations substantially changed beliefs about the regime.

---

# 62. Connection to Information Arrival

This creates an appealing research sequence:

```math
\boxed{
\text{Observation}
\rightarrow
\text{Regime Belief Revision}
\rightarrow
D_{KL}
\rightarrow
\text{Market-State Surprise}.
}
```

This potentially links regime switching directly to the project's information-theoretic architecture.

---

# 63. Regime Surprise as a Predictive Feature

Define

```math
RS_t
=
D_{KL}
\left(
\boldsymbol{\xi}_{t|t}
\Vert
\boldsymbol{\xi}_{t|t-1}
\right).
```

Then test

```math
\boxed{
I
\left(
Y_{t,h};
RS_t
\mid
\mathbf{p}_t
\right)
>0?
}
```

This asks whether sudden changes in inferred market regime contain information about future Bitcoin price adjustment.

---

# 64. Interaction with Latent Information State

Let

```math
\hat{\mathbf{z}}_{t|t}
```

be the continuous latent information state from the state-space model.

The regime model may condition how that state affects price:

```math
P
\left(
Y_{t,h}=1
\mid
\hat{\mathbf{z}}_{t|t},
S_t=j
\right)
=
\sigma
\left(
\beta_{0,j}
+
\boldsymbol{\beta}_j^\top
\hat{\mathbf{z}}_{t|t}
\right).
```

Thus continuous latent state and discrete regime are complementary concepts.

---

# 65. Continuous State Versus Discrete Regime

The latent information state answers:

> What is the current hidden information configuration?

The regime variable answers:

> Which dynamic environment currently governs how that information behaves?

Mathematically:

```math
\boxed{
\mathbf{z}_t
=
\text{continuous hidden state}
}
```

and

```math
\boxed{
S_t
=
\text{discrete hidden regime}.
}
```

---

# 66. Point-in-Time Parameter Estimation

As with every other model in the project, regime parameters cannot be estimated using future holdout observations.

For evaluation fold $k$:

```math
\hat\theta_k
=
f(\mathcal{T}_k).
```

The transition matrix, regime distributions and coefficients are frozen before forecasting

```math
\mathcal{E}_k.
```

Filtering then proceeds forward through evaluation observations.

---

# 67. Online Filtering Is Permitted

During an evaluation period, observing $y_t$ permits updating

```math
P(S_t=j\mid\mathcal{F}_t),
```

because $y_t$ is available at time $t$.

What is prohibited is using future observations

```math
y_{t+1},y_{t+2},\ldots
```

to revise the state estimate at time $t$.

This distinguishes legitimate filtering from leaked smoothing.

---

# 68. Falsifiable Predictions

The regime model generates several testable predictions.

### Prediction 1

Estimated regimes should exhibit meaningful persistence:

```math
p_{ii}
```

should not imply effectively random state assignment unless the data support it.

### Prediction 2

Information coefficients should differ across regimes if information transmission is genuinely state dependent.

### Prediction 3

Order-flow impact may increase in low-liquidity regimes.

### Prediction 4

Information-decay half-lives may differ across states.

### Prediction 5

Regime-conditioned forecasts should outperform a constant-parameter model out of sample.

### Prediction 6

Filtered probabilities—not smoothed probabilities—must retain any forecasting advantage.

---

# 69. What Would Reject the Regime Layer?

The regime model should be removed if:

- states are unstable,
- regimes cannot be economically characterized,
- transition estimates are degenerate,
- information coefficients do not differ meaningfully,
- filtered regimes add no out-of-sample value,
- improvements exist only with smoothed state labels,
- simpler continuous-state models perform equally well.

The architecture does not require a regime model.

---

# 70. Main Mathematical Results

### Markov transition

```math
\boxed{
p_{ij}
=
P
\left(
S_t=j
\mid
S_{t-1}=i
\right)
}
```

### Probability propagation

```math
\boxed{
\boldsymbol{\xi}_{t|t-1}
=
P^\top
\boldsymbol{\xi}_{t-1|t-1}
}
```

### Hamilton update

```math
\boxed{
\xi_{t|t}^{(j)}
=
\frac{
f_j(y_t)
\xi_{t|t-1}^{(j)}
}{
\sum_k
f_k(y_t)
\xi_{t|t-1}^{(k)}
}
}
```

### Expected regime duration

```math
\boxed{
E[D_i]
=
\frac{1}{1-p_{ii}}
}
```

### Regime mixture forecast

```math
\boxed{
p_{t,h}
=
\sum_j
\xi_{t|t}^{(j)}
p_{t,h}^{(j)}
}
```

### Regime entropy

```math
\boxed{
H_t^S
=
-
\sum_j
\xi_{t|t}^{(j)}
\log
\xi_{t|t}^{(j)}
}
```

### Regime surprise

```math
\boxed{
RS_t
=
D_{KL}
\left(
\boldsymbol{\xi}_{t|t}
\Vert
\boldsymbol{\xi}_{t|t-1}
\right)
}
```

---

# 71. Connection to the Main Research Thesis

The regime component now gives the project a formal mechanism for state-dependent information transmission:

```math
\boxed{
\text{Information Arrival}
\rightarrow
\text{Regime Belief Update}
\rightarrow
\text{State-Dependent Information Weight}
\rightarrow
\text{State-Dependent Price Response}
}
```

The central hypothesis becomes

```math
\boxed{
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t,
S_t=j
\right)
}
```

and the key question is whether this quantity varies with $j$.

Equivalently:

```math
\boxed{
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t,
S_t=j
\right)
\neq
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t,
S_t=k
\right).
}
```

This provides the mathematical meaning of **regime-dependent information assimilation** in the project.

---

# 72. Next Mathematical Step

We now have:

```math
\text{Information Theory}
```

```math
\downarrow
```

```math
\text{Microstructure}
```

```math
\downarrow
```

```math
\text{Cointegration / VECM}
```

```math
\downarrow
```

```math
\text{Continuous Latent State}
```

```math
\downarrow
```

```math
\text{Discrete Hidden Regime}.
```

The next mathematical layer is **probabilistic multimodal fusion**.

It will derive how specialist probabilities from:

```math
O_t,
\quad
N_t,
\quad
C_t,
\quad
X_t
```

can be combined coherently through:

- odds,
- log odds,
- Bayes factors,
- conditional independence,
- weighted logit pooling,
- stacking,
- mixture-of-experts models,
- calibration constraints.

This provides the mathematical bridge from separate information domains to the final multimodal predictor.
