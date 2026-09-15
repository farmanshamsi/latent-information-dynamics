# Derivation II — Market Microstructure, Order Flow and Price Formation

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

This document develops the mathematical microstructure foundation of the research programme.

The objective is to connect observable limit-order-book events to price formation through an explicit sequence:

$$
\boxed{
\text{Information}
\rightarrow
\text{Trading Decision}
\rightarrow
\text{Order Flow}
\rightarrow
\text{Liquidity Consumption}
\rightarrow
\text{Quote Revision}
\rightarrow
\text{Price Discovery}
}
$$

The purpose is not to assume that order flow predicts price.

The objective is to derive conditions under which order flow, depth, imbalance and cross-venue liquidity states may contain information about subsequent Bitcoin price adjustment.

---

# 1. Market State

At time $t$, let the best bid be

$$
b_t
$$

and the best ask be

$$
a_t,
$$

with

$$
a_t\geq b_t.
$$

The quoted spread is

$$
s_t
=
a_t-b_t.
$$

The midquote is

$$
m_t
=
\frac{a_t+b_t}{2}.
$$

The midquote provides a cleaner representation of market value than the most recent transaction price because transaction prices mechanically alternate between bid and ask.

---

# 2. Limit-Order-Book State

For depth level

$$
k=1,\ldots,K,
$$

define bid prices and quantities

$$
b_t^{(k)},
\qquad
q_{t,b}^{(k)},
$$

and ask prices and quantities

$$
a_t^{(k)},
\qquad
q_{t,a}^{(k)}.
$$

The complete observed limit-order-book state is

$$
\mathcal{B}_t
=
\left\{
b_t^{(k)},
q_{t,b}^{(k)},
a_t^{(k)},
q_{t,a}^{(k)}
\right\}_{k=1}^{K}.
$$

At the best level,

$$
q_t^B
=
q_{t,b}^{(1)}
$$

and

$$
q_t^A
=
q_{t,a}^{(1)}.
$$

---

# 3. Aggregate Depth

Depth on the bid side through level $K$ is

$$
D_t^B(K)
=
\sum_{k=1}^{K}
q_{t,b}^{(k)}.
$$

Ask depth is

$$
D_t^A(K)
=
\sum_{k=1}^{K}
q_{t,a}^{(k)}.
$$

Total displayed depth is

$$
D_t(K)
=
D_t^B(K)
+
D_t^A(K).
$$

Depth measures how much displayed liquidity is available near the current market price.

---

# 4. Order-Book Imbalance

Define normalized depth imbalance

$$
OBI_t^{(K)}
=
\frac{
D_t^B(K)-D_t^A(K)
}{
D_t^B(K)+D_t^A(K)
}.
$$

Hence

$$
OBI_t^{(K)}
\in[-1,1].
$$

If

$$
OBI_t^{(K)}>0,
$$

displayed bid depth exceeds displayed ask depth.

If

$$
OBI_t^{(K)}<0,
$$

displayed ask depth dominates.

This is a state variable.

It should not automatically be interpreted as informed demand.

---

# 5. Best-Level Imbalance

At the top of the book,

$$
I_t
=
\frac{
q_t^B-q_t^A
}{
q_t^B+q_t^A
}.
$$

This allows the identities

$$
\frac{q_t^B}{q_t^B+q_t^A}
=
\frac{1+I_t}{2},
$$

and

$$
\frac{q_t^A}{q_t^B+q_t^A}
=
\frac{1-I_t}{2}.
$$

These identities will become useful when deriving the microprice.

---

# 6. Stylized Queue-Depletion Model

Suppose that the next quote movement occurs when either:

1. the best ask queue is depleted, producing upward price pressure, or
2. the best bid queue is depleted, producing downward price pressure.

Let

$$
T_A
$$

denote the time until the ask queue is depleted and

$$
T_B
$$

the time until the bid queue is depleted.

For a deliberately stylized model, assume

$$
T_A
\sim
Exp(\lambda_A)
$$

and

$$
T_B
\sim
Exp(\lambda_B).
$$

Assume further that depletion hazard is inversely proportional to displayed queue size:

$$
\lambda_A
=
\frac{\kappa}{q_t^A},
$$

$$
\lambda_B
=
\frac{\kappa}{q_t^B}.
$$

This is not claimed to be an exact empirical law.

It provides a simple probabilistic mechanism linking queue size to directional pressure.

---

# 7. Probability of Ask Depletion First

For independent exponential waiting times,

$$
P(T_A<T_B)
=
\frac{\lambda_A}
{\lambda_A+\lambda_B}.
$$

Substituting

$$
\lambda_A=\frac{\kappa}{q_t^A}
$$

and

$$
\lambda_B=\frac{\kappa}{q_t^B},
$$

gives

$$
P(T_A<T_B)
=
\frac{
1/q_t^A
}{
1/q_t^A+1/q_t^B
}.
$$

Multiplying numerator and denominator by

$$
q_t^Aq_t^B
$$

gives

$$
\boxed{
P(T_A<T_B)
=
\frac{q_t^B}
{q_t^A+q_t^B}.
}
$$

Similarly,

$$
\boxed{
P(T_B<T_A)
=
\frac{q_t^A}
{q_t^A+q_t^B}.
}
$$

Thus greater bid depth increases the stylized probability that the ask side is depleted first.

---

# 8. Derivation of the Microprice

Suppose that after the next queue depletion the one-step reference value is approximately:

$$
a_t
$$

if the ask queue depletes first, and

$$
b_t
$$

if the bid queue depletes first.

Then the conditional expected one-step price is

$$
m_t^{micro}
=
a_tP(T_A<T_B)
+
b_tP(T_B<T_A).
$$

Substituting the depletion probabilities gives

$$
m_t^{micro}
=
a_t
\frac{q_t^B}{q_t^A+q_t^B}
+
b_t
\frac{q_t^A}{q_t^A+q_t^B}.
$$

Therefore

$$
\boxed{
m_t^{micro}
=
\frac{
a_tq_t^B+b_tq_t^A
}{
q_t^A+q_t^B
}.
}
$$

This is the standard volume-weighted microprice form.

---

# 9. Microprice as Midprice Plus Imbalance

Recall

$$
m_t
=
\frac{a_t+b_t}{2}
$$

and

$$
s_t
=
a_t-b_t.
$$

Write

$$
a_t
=
m_t+\frac{s_t}{2},
$$

and

$$
b_t
=
m_t-\frac{s_t}{2}.
$$

Substitute into the microprice:

$$
m_t^{micro}
=
\frac{
\left(m_t+\frac{s_t}{2}\right)q_t^B
+
\left(m_t-\frac{s_t}{2}\right)q_t^A
}{
q_t^A+q_t^B
}.
$$

Collecting terms,

$$
m_t^{micro}
=
m_t
+
\frac{s_t}{2}
\frac{
q_t^B-q_t^A
}{
q_t^B+q_t^A
}.
$$

Therefore

$$
\boxed{
m_t^{micro}
=
m_t
+
\frac{s_t}{2}I_t.
}
$$

Hence the microprice deviation is

$$
\boxed{
m_t^{micro}-m_t
=
\frac{s_t}{2}I_t.
}
$$

This provides a direct mathematical connection between:

$$
\text{spread}
+
\text{queue imbalance}
$$

and a short-horizon directional price estimate.

---

# 10. Interpretation of the Microprice

If

$$
q_t^B=q_t^A,
$$

then

$$
I_t=0
$$

and

$$
m_t^{micro}=m_t.
$$

If

$$
q_t^B>q_t^A,
$$

then

$$
I_t>0
$$

and

$$
m_t^{micro}>m_t.
$$

If

$$
q_t^A>q_t^B,
$$

then

$$
m_t^{micro}<m_t.
$$

The quantity

$$
m_t^{micro}-m_t
$$

can therefore be interpreted as a local order-book pressure measure.

---

# 11. Trades and Aggressor Direction

For transaction $j$, define signed direction

$$
\epsilon_j
=
\begin{cases}
+1, & \text{buyer initiated},\\
-1, & \text{seller initiated}.
\end{cases}
$$

Let trade size be

$$
v_j>0.
$$

Signed traded quantity is

$$
x_j
=
\epsilon_jv_j.
$$

Over interval $\mathcal{T}_t$, signed flow is

$$
Q_t
=
\sum_{j\in\mathcal{T}_t}
\epsilon_jv_j.
$$

---

# 12. Trade Imbalance

Let

$$
V_t^B
=
\sum_{\epsilon_j=+1}v_j,
$$

and

$$
V_t^S
=
\sum_{\epsilon_j=-1}v_j.
$$

Then

$$
Q_t
=
V_t^B-V_t^S.
$$

Total volume is

$$
V_t
=
V_t^B+V_t^S.
$$

Normalized trade imbalance is

$$
TI_t
=
\frac{
V_t^B-V_t^S
}{
V_t^B+V_t^S
}.
$$

Thus

$$
TI_t\in[-1,1].
$$

---

# 13. Why Trade Volume Alone Is Insufficient

Suppose two intervals each contain total volume

$$
V.
$$

In interval A,

$$
V^B=V^S=\frac{V}{2}.
$$

Then

$$
TI=0.
$$

In interval B,

$$
V^B=0.9V
$$

and

$$
V^S=0.1V.
$$

Then

$$
TI
=
\frac{0.9V-0.1V}{V}
=
0.8.
$$

The two intervals have identical total volume but radically different directional pressure.

Therefore

$$
Volume_t
$$

and

$$
SignedFlow_t
$$

measure different aspects of trading activity.

---

# 14. Order-Book Events

Trades alone do not describe the full liquidity process.

Price formation also reflects:

- new limit orders,
- cancellations,
- modifications,
- executions.

Let event $n$ occur at best bid

$$
b_n
$$

with bid quantity

$$
q_n^B,
$$

and best ask

$$
a_n
$$

with ask quantity

$$
q_n^A.
$$

We seek an event-level quantity that assigns positive sign to increased buying pressure and negative sign to increased selling pressure.

---

# 15. Event-Level Order-Flow Imbalance

Define

$$
e_n
=
\mathbf{1}_{\{b_n\geq b_{n-1}\}}q_n^B
-
\mathbf{1}_{\{b_n\leq b_{n-1}\}}q_{n-1}^B
$$

$$
\qquad
-
\mathbf{1}_{\{a_n\leq a_{n-1}\}}q_n^A
+
\mathbf{1}_{\{a_n\geq a_{n-1}\}}q_{n-1}^A.
$$

This expression captures changes in supply and demand at the best quotes.

---

# 16. Bid-Side Interpretation

Consider the bid contribution

$$
e_n^B
=
\mathbf{1}_{\{b_n\geq b_{n-1}\}}q_n^B
-
\mathbf{1}_{\{b_n\leq b_{n-1}\}}q_{n-1}^B.
$$

If

$$
b_n>b_{n-1},
$$

the bid improves and

$$
e_n^B=q_n^B>0.
$$

This represents increased buying pressure.

If

$$
b_n<b_{n-1},
$$

the previous bid disappears and

$$
e_n^B=-q_{n-1}^B<0.
$$

If

$$
b_n=b_{n-1},
$$

both indicators equal one, giving

$$
e_n^B
=
q_n^B-q_{n-1}^B.
$$

Thus quantity added at the bid is positive and quantity removed is negative.

---

# 17. Ask-Side Interpretation

The ask contribution is

$$
e_n^A
=
-
\mathbf{1}_{\{a_n\leq a_{n-1}\}}q_n^A
+
\mathbf{1}_{\{a_n\geq a_{n-1}\}}q_{n-1}^A.
$$

If the ask falls,

$$
a_n<a_{n-1},
$$

new sell liquidity moves closer to the market and contributes negatively.

If the ask rises,

$$
a_n>a_{n-1},
$$

sell-side liquidity has been removed and the contribution is positive.

Thus

$$
e_n
=
e_n^B+e_n^A
$$

summarizes net pressure at the best quotes.

---

# 18. Interval Order-Flow Imbalance

Over interval $k$ containing events

$$
n=N_{k-1}+1,\ldots,N_k,
$$

define

$$
\boxed{
OFI_k
=
\sum_{n=N_{k-1}+1}^{N_k}
e_n.
}
$$

Positive

$$
OFI_k
$$

indicates net pressure toward the ask.

Negative

$$
OFI_k
$$

indicates net pressure toward the bid.

Importantly, OFI includes liquidity additions and removals rather than only executed trades.

---

# 19. From Liquidity to Price Impact

Suppose that near the current price the cumulative quantity available within price distance $\delta$ is approximately linear:

$$
Q(\delta)
\approx
\mathcal{D}_t\delta,
$$

where

$$
\mathcal{D}_t
$$

is local depth density measured in quantity per unit price.

If net aggressive demand is

$$
q,
$$

then the price must move sufficiently far to absorb that quantity.

Set

$$
q
=
\mathcal{D}_t\delta.
$$

Solving for $\delta$ gives

$$
\boxed{
\delta
=
\frac{q}{\mathcal{D}_t}.
}
$$

This provides the simplest mathematical reason that price impact should be inversely related to liquidity.

---

# 20. Linear Impact Approximation

Replacing net aggressive demand with order-flow imbalance gives the local approximation

$$
\Delta m_t
\approx
\beta_t OFI_t
+
\varepsilon_t.
$$

From the depth argument,

$$
\beta_t
\propto
\frac{1}{\mathcal{D}_t}.
$$

Hence a parsimonious specification is

$$
\boxed{
\Delta m_t
=
\frac{\kappa}{\mathcal{D}_t}
OFI_t
+
\varepsilon_t.
}
$$

The relationship is not assumed to be globally linear.

It is a local first-order approximation.

---

# 21. Taylor Interpretation

Suppose price response is generated by nonlinear function

$$
\Delta m
=
f(OFI,D).
$$

Expand around

$$
OFI=0
$$

for fixed depth $D$:

$$
f(OFI,D)
\approx
f(0,D)
+
\left.
\frac{\partial f}{\partial OFI}
\right|_{OFI=0}
OFI.
$$

If

$$
f(0,D)=0,
$$

then

$$
\Delta m
\approx
\beta(D)OFI.
$$

Thus the linear impact equation can be interpreted as a first-order Taylor approximation rather than an assumption of globally linear markets.

---

# 22. Depth-Adjusted OFI

Define

$$
DOFI_t
=
\frac{OFI_t}{D_t},
$$

where $D_t$ is an appropriate local depth measure.

Then

$$
\Delta m_t
=
\kappa DOFI_t+\varepsilon_t
$$

becomes a natural empirical specification.

This suggests that identical order flow may produce very different price responses in different liquidity regimes.

---

# 23. Elasticity Form

A more flexible specification is

$$
|\Delta m_t|
=
c
\frac{
|OFI_t|^\alpha
}{
D_t^\gamma
}
+
\varepsilon_t.
$$

Taking logarithms for positive observations,

$$
\log|\Delta m_t|
=
\log c
+
\alpha\log|OFI_t|
-
\gamma\log D_t
+
u_t.
$$

If

$$
\alpha<1,
$$

price impact is concave in order flow.

If

$$
\gamma>0,
$$

higher depth reduces price impact.

These parameters become empirically estimable quantities.

---

# 24. Kyle Model — Setup

The previous relationship is mechanical.

A deeper explanation of price impact comes from asymmetric information.

Let the terminal asset value be

$$
v
\sim
\mathcal{N}(p_0,\Sigma_0).
$$

An informed trader observes $v$.

A noise trader submits

$$
u
\sim
\mathcal{N}(0,\sigma_u^2)
$$

independently of $v$.

The informed trader submits quantity

$$
x.
$$

The market maker observes only total order flow

$$
y=x+u.
$$

The market maker cannot directly distinguish informed trading from noise trading.

---

# 25. Linear Pricing Rule

Assume the market maker uses

$$
p
=
p_0+\lambda y.
$$

The coefficient

$$
\lambda>0
$$

is Kyle's price-impact coefficient.

Its units are

$$
\frac{\text{price}}
{\text{quantity}}.
$$

Large $\lambda$ means that relatively little order flow moves price substantially.

Therefore

$$
\lambda
$$

is naturally interpreted as an inverse liquidity measure.

---

# 26. Informed Trader's Optimization

Conditional on observing $v$, expected profit is

$$
\pi
=
(v-p)x.
$$

Since

$$
p
=
p_0+\lambda(x+u)
$$

and

$$
E[u]=0,
$$

expected profit conditional on $v$ is

$$
E[\pi\mid v]
=
(v-p_0-\lambda x)x.
$$

Expand:

$$
E[\pi\mid v]
=
(v-p_0)x
-
\lambda x^2.
$$

Differentiate with respect to $x$:

$$
\frac{\partial E[\pi\mid v]}{\partial x}
=
v-p_0-2\lambda x.
$$

Set the first-order condition equal to zero:

$$
v-p_0-2\lambda x=0.
$$

Therefore

$$
\boxed{
x^*
=
\frac{v-p_0}{2\lambda}.
}
$$

The second derivative is

$$
\frac{\partial^2E[\pi\mid v]}{\partial x^2}
=
-2\lambda<0,
$$

confirming a maximum when

$$
\lambda>0.
$$

---

# 27. Informed Trading Intensity

Write informed demand as

$$
x
=
\beta(v-p_0).
$$

Comparing with the optimal strategy gives

$$
\boxed{
\beta
=
\frac{1}{2\lambda}.
}
$$

The informed trader trades more aggressively when market impact is lower.

---

# 28. Market Maker's Bayesian Pricing Rule

Because the variables are jointly Gaussian,

$$
E[v\mid y]
$$

is linear.

Competitive pricing implies

$$
p
=
E[v\mid y].
$$

Therefore

$$
p
=
p_0
+
\frac{
Cov(v,y)
}{
Var(y)
}
y.
$$

Hence

$$
\lambda
=
\frac{
Cov(v,y)
}{
Var(y)
}.
$$

---

# 29. Covariance Between Value and Order Flow

Because

$$
y
=
\beta(v-p_0)+u,
$$

and $u$ is independent of $v$,

$$
Cov(v,y)
=
\beta Var(v).
$$

Therefore

$$
Cov(v,y)
=
\beta\Sigma_0.
$$

Likewise,

$$
Var(y)
=
\beta^2\Sigma_0+\sigma_u^2.
$$

Hence

$$
\boxed{
\lambda
=
\frac{
\beta\Sigma_0
}{
\beta^2\Sigma_0+\sigma_u^2
}.
}
$$

---

# 30. Solving the Kyle Equilibrium

From the informed trader's optimization,

$$
\beta
=
\frac{1}{2\lambda}.
$$

Substitute this into the market maker's equation:

$$
\lambda
=
\frac{
\frac{1}{2\lambda}\Sigma_0
}{
\frac{1}{4\lambda^2}\Sigma_0+\sigma_u^2
}.
$$

Multiply both sides by the denominator:

$$
\frac{\Sigma_0}{4\lambda}
+
\lambda\sigma_u^2
=
\frac{\Sigma_0}{2\lambda}.
$$

Therefore

$$
\lambda\sigma_u^2
=
\frac{\Sigma_0}{4\lambda}.
$$

Multiply by $\lambda$:

$$
\lambda^2\sigma_u^2
=
\frac{\Sigma_0}{4}.
$$

Thus

$$
\boxed{
\lambda
=
\frac{1}{2}
\sqrt{
\frac{\Sigma_0}{\sigma_u^2}
}.
}
$$

Taking the positive root is required because price impact must be positive.

---

# 31. Equilibrium Trading Intensity

Using

$$
\beta
=
\frac{1}{2\lambda},
$$

we obtain

$$
\boxed{
\beta
=
\sqrt{
\frac{\sigma_u^2}{\Sigma_0}
}.
}
$$

Thus the equilibrium links:

$$
\text{fundamental uncertainty},
$$

$$
\text{noise-trader activity},
$$

$$
\text{informed demand},
$$

and

$$
\text{price impact}.
$$

---

# 32. Interpretation of Kyle's Lambda

From

$$
\lambda
=
\frac{1}{2}
\sqrt{
\frac{\Sigma_0}{\sigma_u^2}
},
$$

we obtain

$$
\frac{\partial\lambda}{\partial\Sigma_0}>0.
$$

More uncertainty about fundamental value produces stronger price response to order flow.

Also,

$$
\frac{\partial\lambda}{\partial\sigma_u^2}<0.
$$

Greater noise trading makes a given unit of total order flow less informative.

Thus market impact is not merely a mechanical consequence of quantity.

It can also reflect inference about information.

---

# 33. Predictable Versus Unexpected Order Flow

A critical distinction is between

$$
OFI_t
$$

and the unexpected component of

$$
OFI_t.
$$

Define

$$
\widehat{OFI}_t
=
E
\left[
OFI_t
\mid
\mathcal{F}_{t^-}
\right].
$$

The order-flow innovation is

$$
\boxed{
\xi_t
=
OFI_t-\widehat{OFI}_t.
}
$$

By construction,

$$
E
\left[
\xi_t
\mid
\mathcal{F}_{t^-}
\right]
=
0.
$$

If market participants already expect order flow, that predictable component should contain less new information than the innovation.

---

# 34. Autocorrelated Signed Flow

Suppose signed flow follows

$$
OFI_t
=
\rho OFI_{t-1}
+
\epsilon_t,
$$

where

$$
|\rho|<1
$$

and

$$
E[\epsilon_t\mid\mathcal{F}_{t-1}]=0.
$$

Then

$$
E[OFI_t\mid\mathcal{F}_{t-1}]
=
\rho OFI_{t-1}.
$$

Hence the unexpected component is

$$
\boxed{
\xi_t
=
OFI_t-\rho OFI_{t-1}
=
\epsilon_t.
}
$$

This matters because persistent order flow can arise from order splitting and need not represent repeated arrival of new information.

---

# 35. Information-Based Price Revision

A richer price-update equation is therefore

$$
\boxed{
\Delta m_t^*
=
\lambda_t\xi_t
+
\eta_t,
}
$$

where

$$
m_t^*
$$

is the latent efficient price,

$$
\xi_t
$$

is unexpected order flow,

and

$$
\eta_t
$$

represents public-information innovations.

This distinguishes

$$
\text{raw flow}
$$

from

$$
\text{new information in flow}.
$$

That distinction is central to this research project.

---

# 36. A Bayesian Adverse-Selection Spread Model

We can derive a simple adverse-selection spread from first principles.

Let fundamental value take two possible states:

$$
V
=
\begin{cases}
\mu+\delta,&H,\\
\mu-\delta,&L,
\end{cases}
$$

with equal priors

$$
P(H)=P(L)=\frac{1}{2}.
$$

Suppose a trader is informed with probability

$$
\alpha.
$$

An informed trader:

- buys if the state is $H$,
- sells if the state is $L$.

An uninformed trader buys or sells with equal probability.

---

# 37. Probability of a Buy Given High Value

Conditional on high value,

$$
P(Buy\mid H)
=
\alpha
+
(1-\alpha)\frac{1}{2}.
$$

Therefore

$$
P(Buy\mid H)
=
\frac{1+\alpha}{2}.
$$

Similarly,

$$
P(Buy\mid L)
=
\frac{1-\alpha}{2}.
$$

The unconditional probability of a buy is

$$
P(Buy)
=
\frac{1}{2}.
$$

---

# 38. Posterior Probability After a Buy

Using Bayes' rule,

$$
P(H\mid Buy)
=
\frac{
P(Buy\mid H)P(H)
}{
P(Buy)
}.
$$

Substitute:

$$
P(H\mid Buy)
=
\frac{
\frac{1+\alpha}{2}\frac{1}{2}
}{
\frac{1}{2}
}.
$$

Hence

$$
\boxed{
P(H\mid Buy)
=
\frac{1+\alpha}{2}.
}
$$

Likewise,

$$
\boxed{
P(L\mid Buy)
=
\frac{1-\alpha}{2}.
}
$$

---

# 39. Competitive Ask Price

A competitive liquidity provider sets the ask equal to expected fundamental value conditional on receiving a buy:

$$
a
=
E[V\mid Buy].
$$

Therefore

$$
a
=
(\mu+\delta)
\frac{1+\alpha}{2}
+
(\mu-\delta)
\frac{1-\alpha}{2}.
$$

Expanding gives

$$
a
=
\mu+\alpha\delta.
$$

Hence

$$
\boxed{
a
=
\mu+\alpha\delta.
}
$$

---

# 40. Competitive Bid Price

By symmetry,

$$
b
=
E[V\mid Sell]
$$

gives

$$
\boxed{
b
=
\mu-\alpha\delta.
}
$$

The adverse-selection spread is therefore

$$
s
=
a-b.
$$

Thus

$$
\boxed{
s
=
2\alpha\delta.
}
$$

---

# 41. Interpretation of the Spread Derivation

The simple model implies

$$
\frac{\partial s}{\partial\alpha}>0
$$

and

$$
\frac{\partial s}{\partial\delta}>0.
$$

The spread increases when:

1. informed trading becomes more likely, or
2. uncertainty about true value increases.

Thus spread itself may contain information about market makers' perceived adverse-selection risk.

---

# 42. Observed Price Versus Efficient Price

Let

$$
m_t^*
$$

denote latent efficient price.

A transaction price may be represented schematically as

$$
p_t
=
m_t^*
+
c_tq_t
+
\nu_t,
$$

where

$$
q_t\in\{-1,+1\}
$$

is trade direction,

$$
c_t
$$

captures bid-ask or liquidity effects,

and

$$
\nu_t
$$

captures additional microstructure noise.

Therefore observed transaction-price changes contain both:

$$
\text{efficient-price innovations}
$$

and

$$
\text{microstructure effects}.
$$

This motivates using quotes, order flow and book state rather than treating every transaction-price movement as fundamental information.

---

# 43. Temporary and Permanent Price Impact

Suppose a signed trade shock $x_t$ generates response kernel

$$
G(k),
$$

where $k$ is elapsed event time.

The price process can be represented schematically as

$$
m_t
=
m_0
+
\sum_{s\leq t}
G(t-s)x_s
+
\varepsilon_t.
$$

If

$$
\lim_{k\rightarrow\infty}G(k)
=
G_\infty>0,
$$

then part of the price impact is permanent.

Define

$$
G^{perm}=G_\infty.
$$

The temporary component is

$$
G^{temp}(k)
=
G(k)-G_\infty.
$$

If

$$
G^{temp}(k)\rightarrow0,
$$

the temporary portion eventually decays.

---

# 44. Financial Interpretation

Permanent impact is more consistent with

$$
\text{price discovery},
$$

while transitory impact may arise from

$$
\text{liquidity consumption},
$$

$$
\text{inventory effects},
$$

or

$$
\text{temporary order-book displacement}.
$$

This distinction matters because the project seeks information that predicts subsequent price formation rather than mechanical bid-ask effects alone.

---

# 45. Price and Trade VAR

Let

$$
r_t
=
\Delta\log m_t
$$

and let

$$
q_t
$$

represent signed trade flow or an order-flow variable.

Define

$$
z_t
=
\begin{bmatrix}
r_t\\
q_t
\end{bmatrix}.
$$

A vector autoregression is

$$
\boxed{
z_t
=
\sum_{i=1}^{p}
A_i z_{t-i}
+
u_t.
}
$$

The innovation vector is

$$
u_t
=
\begin{bmatrix}
u_{r,t}\\
u_{q,t}
\end{bmatrix}.
$$

---

# 46. Moving-Average Representation

Assuming the VAR is stable,

$$
A(L)z_t=u_t,
$$

where

$$
A(L)
=
I-A_1L-\cdots-A_pL^p.
$$

Invert the lag polynomial:

$$
z_t
=
A(L)^{-1}u_t.
$$

Define

$$
\Psi(L)
=
A(L)^{-1}.
$$

Then

$$
\boxed{
z_t
=
\sum_{k=0}^{\infty}
\Psi_k u_{t-k}.
}
$$

This is the vector moving-average representation.

---

# 47. Return Response to Order-Flow Innovation

Let

$$
e_r
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

and

$$
e_q
=
\begin{bmatrix}
0\\
1
\end{bmatrix}.
$$

The return response $k$ periods after a unit order-flow innovation is

$$
IRF_{r\leftarrow q}(k)
=
e_r^\top
\Psi_k
e_q.
$$

This is the impulse-response function from unexpected order flow to return.

---

# 48. Permanent Price Impact from the VAR

Since log price is the cumulative sum of returns, the long-run log-price impact of a unit trade innovation is

$$
\Lambda
=
\sum_{k=0}^{\infty}
IRF_{r\leftarrow q}(k).
$$

Therefore

$$
\boxed{
\Lambda
=
\sum_{k=0}^{\infty}
e_r^\top
\Psi_k
e_q.
}
$$

If

$$
\Lambda\neq0,
$$

unexpected trade flow has a permanent long-run effect on the estimated price level.

This provides a statistical measure of the information content of trades.

---

# 49. Why Innovations Matter

Suppose raw trade direction is positively autocorrelated.

Then observing

$$
q_t=+1
$$

after many previous buys may be unsurprising.

A VAR conditions on the predictable component of order flow.

The innovation

$$
u_{q,t}
$$

therefore measures the unexpected component.

The response

$$
IRF_{r\leftarrow q}(k)
$$

is much closer to the research concept of information arrival than a simple contemporaneous correlation between signed trades and returns.

---

# 50. Liquidity Resilience

Liquidity is not only depth at one moment.

It also concerns how quickly the book recovers after a shock.

Let

$$
D_t
$$

denote depth and let long-run normal depth be

$$
\bar D.
$$

Define deviation

$$
Z_t
=
D_t-\bar D.
$$

Suppose

$$
Z_{t+\Delta}
=
\phi Z_t+\epsilon_{t+\Delta},
$$

with

$$
|\phi|<1.
$$

Then

$$
E[Z_{t+n\Delta}\mid Z_t]
=
\phi^nZ_t.
$$

---

# 51. Depth-Recovery Half-Life

Define half-life $h_{1/2}$ by

$$
|\phi|^n
=
\frac{1}{2}.
$$

Taking logs,

$$
n\log|\phi|
=
-\log2.
$$

Therefore

$$
n
=
-\frac{\log2}{\log|\phi|}.
$$

Since each period has duration $\Delta$,

$$
\boxed{
h_{1/2}
=
-\frac{
\Delta\log2
}{
\log|\phi|
}.
}
$$

This gives a quantitative measure of order-book resilience.

---

# 52. Continuous-Time Resilience

The continuous-time analogue is an Ornstein-Uhlenbeck process:

$$
dZ_t
=
-\kappa Z_tdt
+
\sigma dW_t.
$$

Taking conditional expectation gives

$$
E[Z_{t+h}\mid Z_t]
=
Z_te^{-\kappa h}.
$$

The half-life solves

$$
e^{-\kappa h_{1/2}}
=
\frac{1}{2}.
$$

Hence

$$
\boxed{
h_{1/2}
=
\frac{\ln2}{\kappa}.
}
$$

Higher $\kappa$ implies faster liquidity recovery.

---

# 53. Cancellation Intensity

Let

$$
N_t^{cancel,B}
$$

and

$$
N_t^{cancel,A}
$$

denote bid- and ask-side cancellation counts over interval $\Delta$.

Define intensities

$$
\lambda_t^{cancel,B}
=
\frac{
N_t^{cancel,B}
}{
\Delta
},
$$

and

$$
\lambda_t^{cancel,A}
=
\frac{
N_t^{cancel,A}
}{
\Delta
}.
$$

A cancellation imbalance may be defined as

$$
CI_t
=
\frac{
\lambda_t^{cancel,A}
-
\lambda_t^{cancel,B}
}{
\lambda_t^{cancel,A}
+
\lambda_t^{cancel,B}
}.
$$

Large ask-side cancellation relative to bid-side cancellation can remove sell liquidity even before transactions occur.

Thus cancellations may contain pre-trade information.

---

# 54. Replenishment

Suppose a market order removes quantity

$$
Q_t^{removed}
$$

from a queue.

Let newly posted quantity during the next interval be

$$
Q_{t,t+\Delta}^{new}.
$$

Define replenishment ratio

$$
RR_t
=
\frac{
Q_{t,t+\Delta}^{new}
}{
Q_t^{removed}
}.
$$

Values

$$
RR_t>1
$$

indicate strong replenishment.

Values

$$
RR_t<1
$$

indicate incomplete recovery.

This creates a measurable representation of liquidity resilience.

---

# 55. Cross-Venue Price State

Bitcoin trades simultaneously on multiple venues.

For venue $j$, let midprice be

$$
m_t^{(j)}.
$$

For venues $i$ and $j$, define log-price dislocation

$$
d_t^{(i,j)}
=
\log m_t^{(i)}
-
\log m_t^{(j)}.
$$

If the markets represent the same underlying asset and are economically connected, sufficiently large dislocations should face convergence pressure.

---

# 56. Cross-Venue Adjustment

A simple adjustment equation is

$$
\Delta\log m_{t+h}^{(i)}
=
\alpha_i
d_t^{(i,j)}
+
\varepsilon_{t+h}^{(i)}.
$$

If venue $i$ is relatively expensive,

$$
d_t^{(i,j)}>0.
$$

Convergence through downward adjustment at venue $i$ requires

$$
\alpha_i<0.
$$

Alternatively venue $j$ may adjust upward.

Determining which venue adjusts first is a price-discovery question.

The full cointegration treatment is developed separately in the next mathematical derivation.

---

# 57. Cross-Venue Order-Flow Leadership

Price dislocation alone may not identify the information-leading venue.

Suppose

$$
OFI_t^{(i)}
$$

is order flow on venue $i$.

We can test

$$
\Delta m_{t+h}^{(j)}
=
\alpha
+
\beta OFI_t^{(i)}
+
\Gamma^\top X_t
+
\varepsilon_{t+h}.
$$

If

$$
\beta\neq0
$$

after controlling for contemporaneous information on venue $j$, order flow on venue $i$ may contain incremental information about subsequent adjustment on venue $j$.

This is particularly relevant in fragmented cryptocurrency markets.

---

# 58. Microstructure State Vector

The empirical microstructure state can therefore be defined as

$$
\mathbf{o}_t
=
\begin{bmatrix}
OBI_t\\
OFI_t\\
TI_t\\
m_t^{micro}-m_t\\
s_t\\
D_t\\
CI_t\\
RR_t\\
\widehat{\lambda}_t\\
Resilience_t\\
CrossVenue_t
\end{bmatrix}.
$$

This is substantially richer than historical returns alone.

---

# 59. Conditional Expected Price Adjustment

A parsimonious empirical specification is

$$
E
\left[
R_{t,h}
\mid
\mathcal{F}_t
\right]
=
\beta_0
+
\beta_1
\frac{OFI_t}{D_t}
+
\beta_2
\frac{
m_t^{micro}-m_t
}{
s_t
}
+
\beta_3TI_t
$$

$$
+
\beta_4CI_t
+
\beta_5Resilience_t
+
\beta_6CrossVenue_t.
$$

This is not asserted as the true structural model.

It provides a transparent empirical bridge between the mathematical derivations and the prediction system.

---

# 60. Normalized Microprice Signal

From

$$
m_t^{micro}-m_t
=
\frac{s_t}{2}I_t,
$$

we obtain

$$
\frac{
m_t^{micro}-m_t
}{
s_t
}
=
\frac{I_t}{2}.
$$

Thus raw top-level microprice displacement and top-level book imbalance contain equivalent information when spread is used as the normalization.

This warns against treating highly related features as independent discoveries.

---

# 61. Probabilistic Direction Model

For binary target

$$
Y_{t,h}
=
\mathbf{1}
\{R_{t,h}>0\},
$$

define

$$
P
\left(
Y_{t,h}=1
\mid
\mathbf{o}_t,\mathbf{p}_t
\right)
=
\sigma
\left(
\beta_0
+
\boldsymbol{\beta}_O^\top\mathbf{o}_t
+
\boldsymbol{\beta}_P^\top\mathbf{p}_t
\right).
$$

The logistic function is

$$
\sigma(z)
=
\frac{1}{1+e^{-z}}.
$$

The important comparison is not whether

$$
\boldsymbol{\beta}_O
$$

contains individually significant coefficients.

It is whether adding the microstructure state improves genuine out-of-sample probabilistic forecasts.

---

# 62. Information-Theoretic Microstructure Hypothesis

The previous derivation connects directly to the project's information-theoretic foundation.

The microstructure hypothesis is

$$
\boxed{
I
\left(
Y_{t,h};
\mathbf{o}_t
\mid
\mathbf{p}_t
\right)
>0.
}
$$

Under optimal log loss,

$$
I
\left(
Y_{t,h};
\mathbf{o}_t
\mid
\mathbf{p}_t
\right)
=
\mathcal{L}_P^*
-
\mathcal{L}_{P+O}^*.
$$

Thus the microstructure experiment asks whether the mathematical state developed in this document contains information about future Bitcoin movement not already contained in historical price.

---

# 63. Stronger Innovation Hypothesis

A more precise version replaces raw order flow with its innovation:

$$
\xi_t
=
OFI_t
-
E[OFI_t\mid\mathcal{F}_{t^-}].
$$

Then test

$$
\boxed{
I
\left(
Y_{t,h};
\xi_t
\mid
\mathbf{p}_t
\right)
>0.
}
$$

This asks whether **unexpected** liquidity pressure carries incremental information.

It is theoretically stronger than simply testing raw order-flow imbalance.

---

# 64. Depth-Conditioned Hypothesis

Price impact should increase when depth is low.

Therefore define

$$
X_t
=
\frac{\xi_t}{D_t}.
$$

The conditional hypothesis becomes

$$
I
\left(
Y_{t,h};
X_t
\mid
\mathbf{p}_t
\right)
>0.
$$

A related test is whether

$$
\left|
\frac{\partial\Delta m}
{\partial OFI}
\right|
$$

increases as

$$
D_t
$$

falls.

---

# 65. Information-Horizon Hypothesis

Microstructure information should plausibly be strongest at short horizons.

Define

$$
G_O(h)
=
I
\left(
Y_{t,h};
\mathbf{o}_t
\mid
\mathbf{p}_t
\right).
$$

The research can estimate the empirical shape

$$
h
\mapsto G_O(h).
$$

A rapid decline in

$$
G_O(h)
$$

would be consistent with rapid assimilation of microstructure information into price.

---

# 66. Microstructure Information Half-Life

If empirical information gain approximately satisfies

$$
G_O(h)
=
G_0e^{-\lambda_Oh},
$$

then

$$
\boxed{
h_{1/2}^{O}
=
\frac{\ln2}{\lambda_O}.
}
$$

This gives an interpretable measure of how quickly microstructure information loses predictive value.

It potentially links the project's information-theoretic and market-microstructure contributions.

---

# 67. Falsifiable Predictions

The mathematical framework generates several testable predictions.

### Prediction 1

Unexpected positive order-flow imbalance should be associated with positive short-horizon price revision:

$$
E[\Delta m_{t+h}\mid\xi_t>0]
>
E[\Delta m_{t+h}\mid\xi_t=0].
$$

### Prediction 2

Impact should be larger under low depth:

$$
\left|
\frac{\partial\Delta m}{\partial OFI}
\right|_{D=D_L}
>
\left|
\frac{\partial\Delta m}{\partial OFI}
\right|_{D=D_H},
$$

where

$$
D_L<D_H.
$$

### Prediction 3

Microprice displacement should contain short-horizon directional information:

$$
m_t^{micro}>m_t
$$

should, conditional on other variables, be associated with greater probability of upward adjustment.

### Prediction 4

Unexpected flow should be more informative than predictable flow.

### Prediction 5

Liquidity states with slow resilience should exhibit more persistent price impact.

### Prediction 6

Order flow from information-leading venues may predict subsequent price adjustment on lagging venues.

All six predictions may fail empirically.

That possibility is part of the research design.

---

# 68. What Would Falsify the Microstructure Thesis?

Evidence against meaningful incremental microstructure information would include

$$
\mathcal{L}
\left(
\mathcal{M}_{P+O}
\right)
\geq
\mathcal{L}
\left(
\mathcal{M}_P
\right)
$$

on the locked holdout.

It would also include apparent relationships that disappear when:

- predictable order flow is removed,
- depth is controlled,
- spread effects are removed,
- timestamps are corrected,
- cross-validation becomes chronological,
- transaction costs are included,
- the experiment is replicated across venues.

---

# 69. Correlation Is Not Price Discovery

A positive relationship

$$
Corr(OFI_t,\Delta m_t)>0
$$

does not establish that OFI contains information about the future.

The relationship may be contemporaneous and mechanical.

The stronger experiment is

$$
I
\left(
R_{t,h};
OFI_t
\mid
P_t
\right)>0
$$

for

$$
h>0,
$$

under point-in-time data construction.

This distinction is essential.

---

# 70. Prediction Is Not Structural Causality

Even if

$$
OFI_t
$$

predicts

$$
R_{t,h},
$$

the result does not by itself establish the causal structure

$$
OFI_t
\rightarrow
R_{t,h}.
$$

Possible latent drivers include:

- common information events,
- cross-venue activity,
- institutional execution,
- news,
- hidden liquidity,
- correlated algorithmic activity.

The project therefore uses the term

$$
\text{predictive information}
$$

unless stronger causal identification is available.

---

# 71. Empirical Estimation Sequence

The microstructure empirical programme should proceed in the following conceptual order:

$$
\text{Book Reconstruction}
$$

$$
\downarrow
$$

$$
\text{Spread / Depth / Imbalance}
$$

$$
\downarrow
$$

$$
\text{Trade Classification}
$$

$$
\downarrow
$$

$$
OFI
$$

$$
\downarrow
$$

$$
\text{Expected versus Unexpected Flow}
$$

$$
\downarrow
$$

$$
\text{Depth-Adjusted Impact}
$$

$$
\downarrow
$$

$$
\text{Microprice}
$$

$$
\downarrow
$$

$$
\text{Resilience}
$$

$$
\downarrow
$$

$$
\text{Cross-Venue Price Discovery}
$$

$$
\downarrow
$$

$$
\mathcal{M}_{O}
$$

$$
\downarrow
$$

$$
\mathcal{M}_{P+O}
\text{ versus }
\mathcal{M}_{P}.
$$

---

# 72. Mathematical Core

The microstructure component of the research can be summarized through five relationships.

### Queue imbalance

$$
\boxed{
m_t^{micro}-m_t
=
\frac{s_t}{2}I_t
}
$$

### Mechanical liquidity impact

$$
\boxed{
\Delta m_t
\approx
\frac{\kappa}{D_t}OFI_t
}
$$

### Kyle information impact

$$
\boxed{
\lambda
=
\frac{1}{2}
\sqrt{
\frac{\Sigma_0}{\sigma_u^2}
}
}
$$

### Unexpected order flow

$$
\boxed{
\xi_t
=
OFI_t
-
E[OFI_t\mid\mathcal{F}_{t^-}]
}
$$

### Permanent impact

$$
\boxed{
\Lambda
=
\sum_{k=0}^{\infty}
e_r^\top\Psi_ke_q
}
$$

Together these describe several distinct mechanisms through which trading activity may reveal information.

---

# 73. Connection to the Main Research Thesis

The project's central chain is now mathematically more explicit:

$$
\boxed{
\text{Latent Information}
\rightarrow
\text{Trading Decisions}
\rightarrow
\text{Unexpected Order Flow}
\rightarrow
\text{Liquidity Interaction}
\rightarrow
\text{Quote Revision}
\rightarrow
\text{Price Discovery}
}
$$

The empirical question is whether observable components of this process exist early enough to improve the forecast

$$
P
\left(
Y_{t,h}=1
\mid
\mathcal{F}_t
\right)
$$

before the relevant price adjustment is complete.

The microstructure hypothesis is therefore not:

> order-book imbalance predicts Bitcoin.

It is the considerably stronger statement:

$$
\boxed{
I
\left(
Y_{t,h};
\mathbf{o}_t
\mid
\mathbf{p}_t
\right)>0
}
$$

for at least some economically meaningful forecast horizons, under strict point-in-time validation.

That hypothesis will eventually be accepted, qualified or rejected by the data.
