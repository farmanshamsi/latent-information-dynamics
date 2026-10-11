# Derivation VIII — Probabilistic Multimodal Information Fusion

## From Specialist Information Channels to a Unified Bitcoin Probability Forecast

The research programme contains heterogeneous information domains:

```math
O_t
```

for market microstructure,

```math
N_t
```

for news,

```math
C_t
```

for blockchain information,

```math
X_t
```

for cross-market information,

and

```math
V_t
```

for volatility and regime state.

Each domain may contain useful information, but the final research question requires a unified probability:

```math
P
\left(
Y_{t,h}=1
\mid
O_t,N_t,C_t,X_t,V_t,P_t
\right).
```

The mathematical problem is therefore not merely feature concatenation.

It is:

```math
\boxed{
\text{How should several noisy, partially redundant information sources be combined into one calibrated posterior probability?}
}
```

The development path is:

```math
\boxed{
\text{Prior Odds}
\rightarrow
\text{Likelihood Ratios}
\rightarrow
\text{Bayes Factors}
\rightarrow
\text{Posterior Odds}
\rightarrow
\text{Log-Odds Pooling}
\rightarrow
\text{Stacking}
\rightarrow
\text{Regime-Dependent Fusion}
}
```

---

# 1. Binary Forecast Target

Let

```math
Y
=
Y_{t,h}
\in
\{0,1\}.
```

Define the unconditional event probability

```math
\pi
=
P(Y=1).
```

The prior odds are

```math
\boxed{
O(Y)
=
\frac{\pi}{1-\pi}.
}
```

---

# 2. Probability and Odds

For probability $p$,

```math
O
=
\frac{p}{1-p}.
```

Solving for $p$ gives

```math
\boxed{
p
=
\frac{O}{1+O}.
}
```

Thus probability and odds contain equivalent information.

Odds become useful because independent evidence combines multiplicatively.

---

# 3. Log Odds

Define the logit function

```math
\boxed{
\operatorname{logit}(p)
=
\log
\frac{p}{1-p}.
}
```

The inverse-logit function is

```math
\boxed{
\sigma(z)
=
\frac{1}{1+e^{-z}}.
}
```

Hence

```math
p
=
\sigma
\left(
\operatorname{logit}(p)
\right).
```

Multiplication of odds becomes addition in log-odds space.

---

# 4. One Information Source

Let $X$ represent one information channel.

Bayes' rule gives

```math
P(Y=1\mid X)
=
\frac{
p(X\mid Y=1)P(Y=1)
}{
p(X)
}.
```

Similarly,

```math
P(Y=0\mid X)
=
\frac{
p(X\mid Y=0)P(Y=0)
}{
p(X)
}.
```

Taking their ratio,

```math
\frac{
P(Y=1\mid X)
}{
P(Y=0\mid X)
}
=
\frac{
p(X\mid Y=1)
}{
p(X\mid Y=0)
}
\frac{
P(Y=1)
}{
P(Y=0)
}.
```

Therefore,

```math
\boxed{
O(Y\mid X)
=
BF_X
\,
O(Y),
}
```

where

```math
\boxed{
BF_X
=
\frac{
p(X\mid Y=1)
}{
p(X\mid Y=0)
}
}
```

is the Bayes factor supplied by information source $X$.

---

# 5. Interpretation of the Bayes Factor

If

```math
BF_X>1,
```

the observation supports $Y=1$.

If

```math
BF_X<1,
```

the observation supports $Y=0$.

If

```math
BF_X=1,
```

the information source does not change the prior odds.

Thus:

```math
\boxed{
\text{Posterior Odds}
=
\text{Prior Odds}
\times
\text{Evidence}.
}
```

---

# 6. Log Bayes Factor

Taking logarithms,

```math
\log O(Y\mid X)
=
\log O(Y)
+
\log BF_X.
```

Equivalently,

```math
\boxed{
\operatorname{logit}
P(Y=1\mid X)
=
\operatorname{logit}(\pi)
+
\log BF_X.
}
```

Thus each information source contributes an additive evidence term in log-odds space.

---

# 7. Several Information Sources

Suppose we observe

```math
X_1,\ldots,X_M.
```

Bayes' rule gives

```math
O
\left(
Y\mid
X_1,\ldots,X_M
\right)
=
O(Y)
\frac{
p(X_1,\ldots,X_M\mid Y=1)
}{
p(X_1,\ldots,X_M\mid Y=0)
}.
```

The difficulty lies in the joint conditional density.

---

# 8. Conditional Independence Assumption

Suppose the information sources are conditionally independent given $Y$:

```math
p
\left(
X_1,\ldots,X_M
\mid
Y
\right)
=
\prod_{j=1}^{M}
p(X_j\mid Y).
```

Then

```math
\frac{
p(X_1,\ldots,X_M\mid Y=1)
}{
p(X_1,\ldots,X_M\mid Y=0)
}
=
\prod_{j=1}^{M}
\frac{
p(X_j\mid Y=1)
}{
p(X_j\mid Y=0)
}.
```

Therefore,

```math
\boxed{
O
\left(
Y\mid
X_1,\ldots,X_M
\right)
=
O(Y)
\prod_{j=1}^{M}
BF_j.
}
```

---

# 9. Additive Log-Evidence Representation

Taking logs gives

```math
\boxed{
\operatorname{logit}
P
\left(
Y=1
\mid
X_1,\ldots,X_M
\right)
=
\operatorname{logit}(\pi)
+
\sum_{j=1}^{M}
\log BF_j.
}
```

This is the cleanest mathematical justification for additive evidence fusion.

---

# 10. Specialist Posterior Probabilities

Suppose specialist model $j$ outputs

```math
p_j
=
P(Y=1\mid X_j).
```

Its posterior odds are

```math
O_j
=
\frac{p_j}{1-p_j}.
```

From the one-source Bayes equation,

```math
O_j
=
O(Y)BF_j.
```

Therefore,

```math
\boxed{
BF_j
=
\frac{
O_j
}{
O(Y)
}.
}
```

---

# 11. Combining Specialist Probabilities Correctly

Substitute the previous result into the multi-source posterior:

```math
O_{joint}
=
O(Y)
\prod_{j=1}^{M}
\frac{
O_j
}{
O(Y)
}.
```

Therefore,

```math
O_{joint}
=
\frac{
\prod_{j=1}^{M}O_j
}{
O(Y)^{M-1}
}.
```

Hence,

```math
\boxed{
O_{joint}
=
\frac{
\prod_{j=1}^{M}
\frac{p_j}{1-p_j}
}{
\left(
\frac{\pi}{1-\pi}
\right)^{M-1}
}.
}
```

This correction is essential.

---

# 12. Why Multiplying Specialist Odds Naively Is Wrong

A specialist posterior already contains the prior.

If we simply calculate

```math
\prod_jO_j,
```

then the prior odds are counted $M$ times.

But the correct posterior should contain the prior only once.

Therefore the correction

```math
O(Y)^{M-1}
```

must be removed.

This gives:

```math
\boxed{
\text{Joint posterior}
\neq
\text{naive product of specialist posteriors}.
}
```

---

# 13. Combined Specialist Logit Formula

Taking logarithms of the corrected odds gives

```math
\log O_{joint}
=
\sum_{j=1}^{M}
\log O_j
-
(M-1)\log O(Y).
```

Therefore,

```math
\boxed{
\operatorname{logit}(p_{joint})
=
\sum_{j=1}^{M}
\operatorname{logit}(p_j)
-
(M-1)
\operatorname{logit}(\pi).
}
```

This result holds under the conditional-independence assumptions and correctly aligned priors.

---

# 14. Equal Prior Example

If

```math
\pi=\frac{1}{2},
```

then

```math
\operatorname{logit}(\pi)=0.
```

Therefore,

```math
\boxed{
\operatorname{logit}(p_{joint})
=
\sum_{j=1}^{M}
\operatorname{logit}(p_j).
}
```

Only in this special prior setting does simple addition of specialist logits require no prior correction.

---

# 15. Numerical Example

Suppose

```math
\pi=0.50
```

and two independent specialists produce

```math
p_1=0.60
```

and

```math
p_2=0.70.
```

Then

```math
O_1
=
\frac{0.60}{0.40}
=
1.5,
```

and

```math
O_2
=
\frac{0.70}{0.30}
\approx
2.333.
```

With prior odds one,

```math
O_{joint}
=
1.5\times2.333
\approx
3.50.
```

Therefore,

```math
p_{joint}
=
\frac{3.50}{4.50}
\approx
0.778.
```

Independent confirming information strengthens the posterior probability.

---

# 16. Conditional Independence Is Strong

For this project, conditional independence is unlikely to hold exactly.

For example:

```math
\text{News}
\rightarrow
\text{Order Flow}
```

means $N_t$ and $O_t$ may remain statistically related even after conditioning on $Y$.

Similarly:

```math
\text{Macro Shock}
\rightarrow
\text{Cross-Market Movement}
\rightarrow
\text{Bitcoin Order Flow}
```

can create dependence across several specialist information channels.

Therefore naive Bayesian evidence multiplication may double-count information.

---

# 17. Redundant Evidence

Suppose two specialists observe nearly the same underlying signal.

Then

```math
p_1
```

and

```math
p_2
```

may both be high because they react to the same event.

Treating them as independent produces excessive posterior confidence.

Schematically,

```math
\boxed{
\text{Correlated Evidence}
+
\text{Independence Assumption}
\rightarrow
\text{Overconfident Posterior}.
}
```

This is one reason calibration must be central to the fusion layer.

---

# 18. Information Redundancy

Let information domains be $X_1$ and $X_2$.

Both may satisfy

```math
I(Y;X_1\mid P)>0
```

and

```math
I(Y;X_2\mid P)>0.
```

But if their predictive information overlaps,

```math
I(Y;X_2\mid P,X_1)
```

may be much smaller.

Thus:

```math
\boxed{
\text{individual information value}
\neq
\text{unique information value}.
}
```

Fusion must account for this redundancy.

---

# 19. Logistic Stacking

Instead of assuming independence, let each specialist produce probability $p_j$.

Transform probabilities to logits:

```math
\ell_j
=
\operatorname{logit}(p_j).
```

A meta-model estimates

```math
\boxed{
\operatorname{logit}(p_{final})
=
\beta_0
+
\sum_{j=1}^{M}
\beta_j
\ell_j.
}
```

Hence,

```math
\boxed{
p_{final}
=
\sigma
\left(
\beta_0
+
\sum_j
\beta_j
\operatorname{logit}(p_j)
\right).
}
```

This is probabilistic stacking.

---

# 20. Connection to Bayesian Fusion

Under exact conditional independence and perfectly specified specialist models, the theoretically implied coefficients are related to unit evidence weights.

In practice, stacking learns

```math
\beta_j
```

from data.

If two specialists are redundant, the meta-model can reduce their effective weights.

If a specialist contributes little incremental information, its weight may approach zero.

Thus stacking relaxes the strict independence assumption.

---

# 21. Weighted Logarithmic Pooling

A general log-opinion pool combines probabilities through

```math
\boxed{
p_{pool}(y)
\propto
\prod_{j=1}^{M}
p_j(y)^{w_j}.
}
```

For binary outcomes,

```math
\frac{
p_{pool}(1)
}{
p_{pool}(0)
}
=
\prod_{j=1}^{M}
\left[
\frac{
p_j(1)
}{
p_j(0)
}
\right]^{w_j}.
```

Taking logarithms,

```math
\boxed{
\operatorname{logit}(p_{pool})
=
\sum_{j=1}^{M}
w_j
\operatorname{logit}(p_j)
+
c,
}
```

where $c$ may incorporate normalization or prior correction.

Thus logistic stacking is closely related to learned logarithmic pooling.

---

# 22. Convex Probability Averaging

A simpler alternative is

```math
\boxed{
p_{final}
=
\sum_{j=1}^{M}
w_jp_j,
}
```

subject to

```math
w_j\geq0
```

and

```math
\sum_jw_j=1.
```

This guarantees

```math
0\leq p_{final}\leq1.
```

It is easy to interpret but has a different mathematical meaning from evidence combination in odds space.

---

# 23. Probability Average Versus Log-Odds Pooling

Consider two specialists:

```math
p_1=0.90,
```

```math
p_2=0.60.
```

Equal probability averaging gives

```math
p_{avg}=0.75.
```

Log-odds pooling treats probability evidence multiplicatively in odds space.

Thus the two approaches encode different assumptions.

Neither should be declared universally superior before empirical testing.

---

# 24. Linear Opinion Pool

The convex average

```math
p_{final}
=
\sum_jw_jp_j
```

is known as a linear opinion pool.

It tends to produce probabilities lying between specialist forecasts.

This can improve robustness when individual specialists are overconfident.

---

# 25. Logarithmic Opinion Pool

The logarithmic pool

```math
p_{final}(y)
\propto
\prod_jp_j(y)^{w_j}
```

can reinforce agreement more strongly.

For binary forecasts this corresponds naturally to weighted combinations of logits.

It can become overconfident if highly correlated specialists receive excessive weight.

---

# 26. Why We Need Out-of-Sample Meta-Training

Suppose specialist model $j$ is trained on observations

```math
1,\ldots,T.
```

If we generate predictions on those same observations and use them to train the meta-model, the meta-model receives in-sample specialist performance.

This creates stacking leakage.

The specialist probabilities used to train the fusion model must themselves be out of sample.

---

# 27. Cross-Fitted Specialist Predictions

Partition the development sample chronologically.

For fold $k$, train specialist $j$ using

```math
\mathcal{T}_k
```

and generate predictions on

```math
\mathcal{E}_k.
```

Collect:

```math
\hat p_{j,t}^{OOF}
```

for each observation $t$.

The meta-model is trained only on these out-of-fold predictions.

Thus its training matrix is

```math
\boxed{
Z_t
=
\begin{bmatrix}
\operatorname{logit}
\hat p_{O,t}^{OOF}\\
\operatorname{logit}
\hat p_{N,t}^{OOF}\\
\operatorname{logit}
\hat p_{C,t}^{OOF}\\
\operatorname{logit}
\hat p_{X,t}^{OOF}
\end{bmatrix}.
}
```

---

# 28. Time-Series Stacking

For financial data, ordinary random K-fold stacking is inappropriate.

The required sequence is:

```math
\boxed{
\text{Past Training}
\rightarrow
\text{Future Specialist Prediction}
\rightarrow
\text{OOF Probability}
\rightarrow
\text{Meta-Model Training}.
}
```

All specialist predictions used for meta-learning must respect chronological ordering.

---

# 29. Final Specialist Retraining

After the meta-model structure is selected using development data, specialists can be refitted using the full permitted training region.

They then generate genuine future predictions:

```math
\hat p_{j,t}.
```

The frozen meta-model combines them.

No final holdout observation may influence either specialist or meta-model fitting.

---

# 30. Specialist Calibration

Suppose specialist probability $p_j$ is systematically overconfident.

Then its logit

```math
\operatorname{logit}(p_j)
```

may have excessive magnitude.

Therefore specialists may require calibration before fusion.

Possible calibration transformations include:

```math
\text{Platt scaling},
```

```math
\text{isotonic calibration},
```

or another training-only calibration model.

Calibration itself must be estimated without holdout leakage.

---

# 31. Logistic Calibration

Let raw model score be $s$.

Platt-style calibration estimates

```math
\boxed{
P(Y=1\mid s)
=
\sigma
\left(
a+bs
\right).
}
```

If the raw score is already a logit, then $b$ acts as a confidence-temperature adjustment.

---

# 32. Temperature Scaling

For model logit $\ell$,

```math
\boxed{
p_T
=
\sigma
\left(
\frac{\ell}{T}
\right).
}
```

If

```math
T>1,
```

extreme probabilities are softened.

If

```math
0<T<1,
```

probabilities become more extreme.

For a binary system with no additional intercept, this is a simple calibration transformation.

---

# 33. Fusion After Calibration

A principled sequence is:

```math
\boxed{
\text{Raw Specialist}
\rightarrow
\text{OOF Prediction}
\rightarrow
\text{Calibration}
\rightarrow
\text{Fusion}
}
```

rather than allowing the meta-model to compensate blindly for severe specialist miscalibration.

Both approaches can be tested empirically.

---

# 34. Meta-Model Likelihood

For logistic stacking,

```math
p_t
=
\sigma
\left(
\beta_0
+
\boldsymbol{\beta}^{\top}
\boldsymbol{\ell}_t
\right).
```

The Bernoulli likelihood is

```math
L(\beta)
=
\prod_{t=1}^{T}
p_t^{Y_t}
(1-p_t)^{1-Y_t}.
```

The log likelihood is

```math
\boxed{
\ell(\beta)
=
\sum_{t=1}^{T}
\left[
Y_t\log p_t
+
(1-Y_t)\log(1-p_t)
\right].
}
```

Maximizing this likelihood is equivalent to minimizing log loss.

---

# 35. Regularized Fusion

If specialist predictions are correlated, the meta-model coefficients can become unstable.

A ridge-regularized objective is

```math
\boxed{
\hat{\beta}
=
\arg\min_{\beta}
\left[
\mathcal{L}_{log}(\beta)
+
\lambda
\|\beta\|_2^2
\right].
}
```

An $\ell_1$ penalty gives

```math
\mathcal{L}_{log}
+
\lambda
\|\beta\|_1.
```

This can perform sparse specialist selection.

Hyperparameter $\lambda$ must be selected chronologically within development data.

---

# 36. Interpreting Fusion Weights

A positive coefficient

```math
\beta_j>0
```

does not automatically mean specialist $j$ is causally important.

It means the specialist's log-odds contain incremental predictive information conditional on the other meta-features under the fitted model.

Likewise,

```math
\beta_j\approx0
```

may indicate:

- little signal,
- redundancy,
- poor calibration,
- instability.

---

# 37. Negative Meta-Weights

It is possible to estimate

```math
\beta_j<0.
```

This means that, conditional on the other specialists, higher probability from specialist $j$ is associated with lower event probability.

Possible explanations include:

- systematic specialist bias,
- redundancy and multicollinearity,
- regime dependence,
- overfitting.

Negative weights require investigation rather than automatic acceptance.

---

# 38. Domain Ablation at the Fusion Layer

Let full fusion model be

```math
\mathcal{M}_{full}.
```

Remove specialist $j$ to obtain

```math
\mathcal{M}_{-j}.
```

Define

```math
\boxed{
\Delta_j
=
\mathcal{L}
\left(
\mathcal{M}_{-j}
\right)
-
\mathcal{L}
\left(
\mathcal{M}_{full}
\right).
}
```

Positive $\Delta_j$ means removing the specialist worsens forecast loss.

This provides a direct domain-level contribution measure.

---

# 39. Specialist Diversity

A useful ensemble requires not only strong specialists but also diversity.

If every specialist makes nearly identical errors,

```math
Corr
\left(
e_i,e_j
\right)
\approx1,
```

then combining them offers little diversification.

Therefore the research should inspect both:

```math
\text{individual predictive skill}
```

and

```math
\text{error correlation}.
```

---

# 40. Error Covariance

Let specialist probability error be

```math
e_{j,t}
=
p_{j,t}
-
Y_t.
```

Define covariance matrix

```math
\Sigma_e
=
E[
\mathbf{e}_t
\mathbf{e}_t^\top
].
```

Strong off-diagonal elements indicate specialists make correlated errors.

This can help explain why adding another apparently accurate specialist produces little incremental gain.

---

# 41. Minimum-Variance Analogy

For unbiased scalar estimators, one classical combination minimizes

```math
w^\top\Sigma_e w
```

subject to

```math
\mathbf{1}^\top w=1.
```

The solution is

```math
\boxed{
w^*
=
\frac{
\Sigma_e^{-1}\mathbf{1}
}{
\mathbf{1}^\top
\Sigma_e^{-1}
\mathbf{1}
}.
}
```

This provides intuition for why correlated specialists should receive less combined weight.

For probability forecasts, proper scoring rules remain preferable to simply applying this estimator directly.

---

# 42. Mixture of Experts

Instead of one fixed set of fusion weights, define specialist probabilities

```math
p_t^{(j)}
```

and gating weights

```math
g_j(\mathbf{s}_t).
```

Require

```math
g_j(\mathbf{s}_t)\geq0
```

and

```math
\sum_jg_j(\mathbf{s}_t)=1.
```

Then

```math
\boxed{
p_t
=
\sum_{j=1}^{M}
g_j(\mathbf{s}_t)
p_t^{(j)}.
}
```

The weights now depend on current state $\mathbf{s}_t$.

---

# 43. Softmax Gating Network

A standard gate is

```math
\boxed{
g_j(\mathbf{s}_t)
=
\frac{
\exp
\left(
\gamma_j^\top\mathbf{s}_t
\right)
}{
\sum_{k=1}^{M}
\exp
\left(
\gamma_k^\top\mathbf{s}_t
\right)
}.
}
```

This guarantees valid convex weights.

Possible gating variables include:

- filtered regime probabilities,
- volatility,
- liquidity,
- latent-state uncertainty.

---

# 44. Regime-Dependent Fusion

Let

```math
\xi_{t|t}^{(k)}
```

be the filtered probability of regime $k$.

Within each regime, specialist fusion may be

```math
\operatorname{logit}
p_t^{(k)}
=
\beta_{0,k}
+
\sum_j
\beta_{j,k}
\operatorname{logit}
p_{j,t}.
```

The final forecast is

```math
\boxed{
p_t
=
\sum_k
\xi_{t|t}^{(k)}
p_t^{(k)}.
}
```

This combines the regime and multimodal layers naturally.

---

# 45. State-Dependent Specialist Relevance

Suppose microstructure is most useful during low liquidity while macro information matters more at longer, calmer horizons.

Then it is possible that

```math
\beta_{O,stress}
>
\beta_{O,normal}
```

while

```math
\beta_{X,stress}
<
\beta_{X,normal}.
```

The model can estimate this rather than imposing constant information weights.

---

# 46. Hierarchical Fusion

The complete architecture may be hierarchical.

First create domain-level states:

```math
O_t
\rightarrow
p_t^O,
```

```math
N_t
\rightarrow
p_t^N,
```

```math
C_t
\rightarrow
p_t^C,
```

```math
X_t
\rightarrow
p_t^X.
```

Then combine them:

```math
\boxed{
(p_t^O,p_t^N,p_t^C,p_t^X)
\rightarrow
p_t^{I}.
}
```

Finally combine information probability with a price specialist:

```math
\boxed{
(p_t^{I},p_t^{P})
\rightarrow
p_t^{I+P}.
}
```

This preserves the central experimental distinction between price and non-price information.

---

# 47. Why Price Should Remain a Separate Specialist

If all price features are mixed immediately with all information features, it becomes harder to answer the central research question.

Keeping

```math
p_t^P
```

separate from

```math
p_t^I
```

allows direct comparison of:

```math
\mathcal{M}_P,
```

```math
\mathcal{M}_I,
```

and

```math
\mathcal{M}_{I+P}.
```

This is scientifically more informative than building only one giant model.

---

# 48. Direct Early Fusion

Early fusion concatenates raw or transformed feature blocks:

```math
\mathbf{x}_t
=
\begin{bmatrix}
\mathbf{o}_t\\
\mathbf{n}_t\\
\mathbf{c}_t\\
\mathbf{x}_t^{cross}\\
\mathbf{v}_t
\end{bmatrix}.
```

One model estimates

```math
\boxed{
p_t
=
f(\mathbf{x}_t).
}
```

This allows interactions to be learned directly.

But it also creates:

- high dimensionality,
- scale differences,
- missing-data problems,
- interpretability challenges.

---

# 49. Late Fusion

Late fusion first produces specialist forecasts:

```math
p_t^O,
\quad
p_t^N,
\quad
p_t^C,
\quad
p_t^X,
\quad
p_t^V.
```

Then the meta-model combines them.

Advantages include:

- modularity,
- domain-level ablation,
- separate calibration,
- asynchronous modelling,
- easier debugging.

The project should compare early and late fusion rather than assuming one dominates.

---

# 50. Hybrid Fusion

A hybrid design can combine both.

For example, domain representations

```math
z_t^O,
\quad
z_t^N,
\quad
z_t^C,
\quad
z_t^X
```

may enter both specialist models and a joint interaction model.

The final system could then fuse:

```math
\text{specialist probabilities}
+
\text{selected cross-domain interactions}.
```

This is more complex and should only be introduced if justified.

---

# 51. Interaction Terms

Suppose news is particularly informative when order-flow response confirms it.

A meta-model may include

```math
\operatorname{logit}(p_N)
\times
\operatorname{logit}(p_O).
```

Then

```math
\operatorname{logit}(p)
=
\beta_0
+
\beta_N\ell_N
+
\beta_O\ell_O
+
\beta_{NO}\ell_N\ell_O.
```

A nonzero $\beta_{NO}$ represents interaction beyond additive specialist evidence.

---

# 52. Synergistic Evidence

Conceptually, synergy exists when the joint information state has more predictive value than separate marginal contributions suggest.

A simple diagnostic is

```math
S_{NO}
=
I(Y;N,O\mid P)
-
I(Y;N\mid P)
-
I(Y;O\mid P).
```

Positive values can indicate complementary information under this simplified comparison.

A rigorous partial-information decomposition is more subtle, so this quantity should be interpreted cautiously.

---

# 53. Forecast Horizon Dependence

Specialist relevance may depend on horizon $h$.

Write

```math
p_{j,t,h}.
```

Then fusion weights may also depend on horizon:

```math
\beta_{j,h}.
```

For example:

```math
\beta_{O,1m}
```

may be large while

```math
\beta_{O,1d}
```

is small.

Conversely, slower information sources may matter more at longer horizons.

---

# 54. Horizon-Specific Fusion

The principled approach is therefore to estimate separate fusion systems:

```math
g_h
```

for each primary horizon.

Thus,

```math
\boxed{
p_{t,h}
=
g_h
\left(
p_{O,t,h},
p_{N,t,h},
p_{C,t,h},
p_{X,t,h},
p_{V,t,h}
\right).
}
```

A single universal fusion rule across all horizons is not assumed.

---

# 55. Missing Specialist Forecasts

At time $t$, some specialists may not have fresh information.

For example, no new news event may exist.

One approach is to include:

```math
Age_{j,t}
```

and

```math
Available_{j,t}
```

alongside the specialist probability.

Another is a gating function that assigns lower weight to stale information.

Thus:

```math
g_j
=
g_j
\left(
\text{state},
\text{age},
\text{availability}
\right).
```

---

# 56. Stale Probability Problem

If a specialist probability has not been updated for many hours, treating it identically to a fresh probability can be misleading.

A decayed log-evidence representation could be

```math
\ell_{j,t}^{decay}
=
e^{-\lambda_jAge_{j,t}}
\ell_{j,t}.
```

This is an optional mechanism.

The decay rate must be estimated without future information.

---

# 57. Information Half-Life and Fusion

From previous derivations, domain $j$ may have estimated information half-life

```math
h_{1/2}^{(j)}.
```

Then

```math
\lambda_j
=
\frac{\ln2}
{h_{1/2}^{(j)}}.
```

This potentially links empirically estimated information decay directly to dynamic fusion weights.

---

# 58. Uncertainty-Weighted Fusion

Suppose specialist $j$ produces both probability

```math
p_j
```

and uncertainty measure

```math
u_j.
```

A gate may reduce weight when uncertainty is high:

```math
g_j
\propto
\exp
\left(
-\gamma u_j
\right).
```

After normalization,

```math
\boxed{
g_j
=
\frac{
e^{-\gamma u_j}
}{
\sum_k
e^{-\gamma u_k}
}.
}
```

This is one possible way to incorporate model uncertainty into fusion.

---

# 59. Latent-State Uncertainty

For the state-space specialist, uncertainty is naturally available through

```math
P_{t|t}.
```

For example,

```math
u_t^{state}
=
\operatorname{tr}
\left(
P_{t|t}
\right).
```

The fusion model could reduce reliance on the latent-state specialist when posterior state uncertainty is unusually high.

---

# 60. Entropy of Specialist Probability

A binary forecast's predictive entropy is

```math
\boxed{
H(p)
=
-p\log p
-
(1-p)\log(1-p).
}
```

Maximum uncertainty occurs at

```math
p=0.5.
```

Extreme probabilities have lower entropy.

However, low entropy is useful only if the specialist is calibrated.

An overconfident bad model has low entropy but poor information quality.

---

# 61. Fusion Calibration

The final probability forecast must itself be calibrated:

```math
P(Y=1\mid p_{final}=q)
\approx
q.
```

Even if every specialist is individually calibrated, their combination is not automatically calibrated.

Therefore calibration must be evaluated after fusion.

---

# 62. Proper Evaluation

The principal fusion metric remains out-of-sample log loss:

```math
\boxed{
\mathcal{L}_{log}
=
-
\frac{1}{N}
\sum_t
\left[
Y_t\log p_t
+
(1-Y_t)\log(1-p_t)
\right].
}
```

Brier score and calibration diagnostics provide secondary evidence.

Directional accuracy alone is insufficient.

---

# 63. Why Log Loss Is Especially Appropriate

Fusion aims to improve the entire probability distribution, not merely classification.

Log loss strongly penalizes confident incorrect forecasts.

Thus it discourages fusion systems that appear accurate only because they create extreme but poorly calibrated probabilities.

---

# 64. Main Experimental Fusion Ladder

The eventual empirical comparison should proceed from simple to complex:

```math
\text{Equal Probability Average}
```

then

```math
\text{Optimized Convex Average}
```

then

```math
\text{Logit Stacking}
```

then, if justified,

```math
\text{Regime-Dependent / State-Dependent Fusion}.
```

More complex fusion survives only if it improves out-of-sample probability forecasts.

---

# 65. Price Versus Information Fusion Test

Let

```math
p_t^P
```

be the price-only forecast and

```math
p_t^I
```

the multimodal non-price information forecast.

The final model may be

```math
\operatorname{logit}
p_t^{I+P}
=
\beta_0
+
\beta_P
\operatorname{logit}
p_t^P
+
\beta_I
\operatorname{logit}
p_t^I.
```

The central empirical question is whether

```math
\beta_I
```

produces genuine out-of-sample improvement beyond $p_t^P$.

---

# 66. Connection to Conditional Information

The theoretical hypothesis is

```math
I(Y;I\mid P)>0.
```

The empirical fusion system approximates this comparison through:

```math
\mathcal{M}_P
```

versus

```math
\mathcal{M}_{I+P}.
```

Under optimal log scoring,

```math
I(Y;I\mid P)
=
\mathcal{L}_P^*
-
\mathcal{L}_{I+P}^*.
```

Thus multimodal fusion is not separate from the project's information-theoretic core.

It is the practical estimator of that information comparison.

---

# 67. Fusion Failure Modes

The fusion layer can fail through:

- correlated specialists,
- miscalibrated probabilities,
- in-sample stacking,
- stale signals,
- regime instability,
- excessive meta-model complexity,
- information duplication,
- holdout tuning.

Each failure can create apparently strong but false predictive confidence.

---

# 68. Falsifiable Predictions

The fusion framework creates several testable predictions.

A multimodal information model should outperform the strongest individual information specialist if the domains contain complementary signal.

Cross-fitted stacking should outperform naive probability multiplication when specialist errors are correlated.

Some domain weights should change across regimes if information transmission is state dependent.

The combined price-plus-information model should outperform the price-only model if non-price information contains incremental signal.

Final probabilities should remain reasonably calibrated.

All of these predictions may fail.

---

# 69. What Would Reject Complex Fusion?

A complex fusion model should be rejected if:

- equal averaging performs equally well,
- specialist weights are unstable,
- improvements disappear out of sample,
- calibration deteriorates,
- one specialist dominates completely,
- apparent improvement comes from stacking leakage,
- regime-dependent weights add no stable value.

Complexity is not itself a contribution.

---

# 70. Main Mathematical Results

### Bayes factor

```math
\boxed{
BF_j
=
\frac{
p(X_j\mid Y=1)
}{
p(X_j\mid Y=0)
}
}
```

### Independent evidence fusion

```math
\boxed{
O
\left(
Y\mid X_1,\ldots,X_M
\right)
=
O(Y)
\prod_{j=1}^{M}
BF_j
}
```

### Specialist-posterior correction

```math
\boxed{
O_{joint}
=
\frac{
\prod_jO_j
}{
O(Y)^{M-1}
}
}
```

### Joint specialist logit

```math
\boxed{
\operatorname{logit}(p_{joint})
=
\sum_j
\operatorname{logit}(p_j)
-
(M-1)
\operatorname{logit}(\pi)
}
```

### Learned logit fusion

```math
\boxed{
\operatorname{logit}(p_{final})
=
\beta_0
+
\sum_j
\beta_j
\operatorname{logit}(p_j)
}
```

### Regime-weighted forecast

```math
\boxed{
p_t
=
\sum_k
\xi_{t|t}^{(k)}
p_t^{(k)}
}
```

---

# 71. Final Research Architecture

The multimodal predictive structure can now be written as

```math
O_t
\rightarrow
p_t^O,
```

```math
N_t
\rightarrow
p_t^N,
```

```math
C_t
\rightarrow
p_t^C,
```

```math
X_t
\rightarrow
p_t^X,
```

followed by

```math
\boxed{
p_t^I
=
g
\left(
p_t^O,
p_t^N,
p_t^C,
p_t^X,
p_t^V
\right).
}
```

The final research model then evaluates

```math
\boxed{
p_t^{I+P}
=
h
\left(
p_t^I,
p_t^P
\right).
}
```

This preserves both modularity and the scientific distinction between price and non-price information.

---

# 72. Connection to the Main Thesis

The full information pathway is now:

```math
\boxed{
\text{Raw Information}
\rightarrow
\text{Domain State}
\rightarrow
\text{Specialist Probability}
\rightarrow
\text{Calibrated Evidence}
\rightarrow
\text{Multimodal Fusion}
\rightarrow
\text{Bitcoin Probability Forecast}.
}
```

The final test remains:

```math
\boxed{
\mathcal{L}
\left(
\mathcal{M}_{I+P}
\right)
<
\mathcal{L}
\left(
\mathcal{M}_{P}
\right)
}
```

under strict point-in-time out-of-sample evaluation.

Only then can the project claim that multimodal non-price information contributes predictive value beyond Bitcoin's own historical price.
