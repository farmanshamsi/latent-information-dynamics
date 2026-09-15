# Data Architecture

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

This document defines the proposed data architecture, timestamp semantics, information domains, synchronization rules and point-in-time controls for the research programme.

The central requirement is:

> A feature may only enter a prediction if that information could genuinely have been observed before the prediction timestamp.

The architecture is therefore designed around **information availability**, not merely historical database timestamps.

---

# 1. Design Principle

The research investigates the mapping

$$
\text{Information Arrival}
\rightarrow
\text{Market State}
\rightarrow
\text{Price Adjustment}.
$$

Consequently, the data system must preserve the temporal ordering of information.

For every observation $x$, distinguish:

$$
t_x^{event}
$$

the time the underlying event occurred,

$$
t_x^{publish}
$$

the time the information was publicly released,

$$
t_x^{receive}
$$

the time the research system could have received it,

and

$$
t_x^{process}
$$

the time the observation became usable after processing.

The usable timestamp is therefore

$$
t_x^{available}
=
\max
\left(
t_x^{publish},
t_x^{receive},
t_x^{process}
\right).
$$

A prediction at time $t$ may use $x$ only if

$$
t_x^{available}\leq t.
$$

This rule takes precedence over the timestamp stored by a vendor or historical API.

---

# 2. Information Domains

The proposed information universe contains six major data domains:

$$
\mathcal{D}
=
\{
D_P,
D_O,
D_N,
D_C,
D_X,
D_V
\},
$$

where:

- $D_P$: Bitcoin price and trade data,
- $D_O$: order-book and order-flow data,
- $D_N$: news and textual information,
- $D_C$: blockchain and crypto-economic information,
- $D_X$: cross-market and macro-financial information,
- $D_V$: derivatives, volatility and regime information.

Each domain will maintain its own native timestamp and update frequency.

---

# 3. Price and Trade Data

The price layer provides both:

1. the prediction target,
2. the deliberately strong price-history benchmark.

Required fields may include:

| Field | Description |
|---|---|
| timestamp | exchange timestamp |
| exchange | trading venue |
| symbol | instrument |
| trade_price | executed price |
| trade_size | executed quantity |
| side | buyer/seller aggressor where available |
| trade_id | venue-specific identifier |
| bid | best bid |
| ask | best ask |

The primary target instrument will initially be Bitcoin against a major quote currency or stablecoin.

The exact venue and contract will be fixed before empirical work begins.

---

# 4. Target Construction

For reference price $S_t$, define forward return

$$
R_{t,h}
=
\log
\left(
\frac{S_{t+h}}{S_t}
\right).
$$

Binary direction is

$$
Y_{t,h}
=
\mathbf{1}
\left\{
R_{t,h}>\tau_h
\right\}.
$$

Potential horizons may include:

$$
h\in
\{
1\text{ min},
5\text{ min},
15\text{ min},
1\text{ hour},
4\text{ hours},
1\text{ day}
\}.
$$

These are candidate horizons, not fixed final specifications.

The final horizon set will be selected before final out-of-sample testing.

---

# 5. Reference Price

Using last traded price can introduce microstructure noise.

Candidate reference prices include:

## Midquote

$$
m_t
=
\frac{a_t+b_t}{2}.
$$

## Microprice

A depth-weighted top-of-book approximation may be written

$$
m_t^{micro}
=
\frac{
a_t q_t^{B}
+
b_t q_t^{A}
}{
q_t^{A}+q_t^{B}
}.
$$

The target construction will compare appropriate reference-price definitions before the test set is opened.

---

# 6. Limit-Order-Book Data

For depth level $k$, record

$$
b_t^{(k)},
\quad
q_{t,b}^{(k)},
\quad
a_t^{(k)},
\quad
q_{t,a}^{(k)}.
$$

A minimal event schema is:

| Field | Description |
|---|---|
| exchange_timestamp | exchange event time |
| receive_timestamp | local/system receipt time |
| exchange | venue |
| symbol | instrument |
| update_id | sequence identifier |
| side | bid / ask |
| price | order-book level |
| quantity | displayed quantity |
| event_type | insert / modify / delete |

The sequence identifier is important for reconstructing the book correctly.

---

# 7. Order-Book Features

Potential microstructure variables include:

## Spread

$$
Spread_t
=
a_t-b_t.
$$

## Relative spread

$$
RelSpread_t
=
\frac{a_t-b_t}{m_t}.
$$

## Depth imbalance

$$
OBI_t^{(K)}
=
\frac{
\sum_{k=1}^{K}q_{t,b}^{(k)}
-
\sum_{k=1}^{K}q_{t,a}^{(k)}
}{
\sum_{k=1}^{K}q_{t,b}^{(k)}
+
\sum_{k=1}^{K}q_{t,a}^{(k)}
}.
$$

## Signed trade imbalance

$$
TI_t
=
\frac{
V_t^{buy}-V_t^{sell}
}{
V_t^{buy}+V_t^{sell}
}.
$$

Additional candidates include:

- order-flow imbalance,
- cancellation intensity,
- queue depletion,
- depth slope,
- liquidity concentration,
- replenishment rate,
- resilience,
- realized price impact.

These variables will be computed from historical events rather than future-completed bars.

---

# 8. Event-Time Versus Clock-Time Representation

A central design choice is whether observations should be sampled in:

$$
\text{clock time}
$$

or

$$
\text{event time}.
$$

Clock-time examples include:

- one-second intervals,
- one-minute intervals,
- five-minute intervals.

Event-time alternatives include:

- fixed trade counts,
- fixed volume,
- fixed order-book events,
- information arrivals.

The project may maintain both representations.

Raw events remain in event time.

Model-ready datasets may subsequently be constructed in clock time or event time depending on the hypothesis.

---

# 9. News Data

For each article or information event $j$, the desired raw schema includes:

| Field | Description |
|---|---|
| event_id | unique identifier |
| source | publisher |
| title | headline |
| body | text where available |
| publish_time | original publication timestamp |
| receive_time | acquisition timestamp |
| update_time | article revision timestamp |
| entities | detected entities |
| url_hash | deduplication identifier |

The distinction between publication and acquisition is essential.

A historical news database must not be treated as if every article were instantaneously available.

---

# 10. News-Derived Features

The NLP layer may generate:

$$
\mathbf{e}_j
=
\phi(x_j)
$$

for text embedding $\mathbf{e}_j$.

Structured features may include:

$$
Sentiment_j,
$$

$$
Novelty_j,
$$

$$
Uncertainty_j,
$$

$$
Relevance_j,
$$

$$
EventType_j.
$$

The model-ready news state at time $t$ must only aggregate events satisfying

$$
t_j^{available}\leq t.
$$

An example decay model is

$$
w_j(t)
=
e^{-\lambda(t-t_j^{available})}.
$$

Then

$$
\mathbf{n}_t
=
\sum_{j:t_j^{available}\leq t}
w_j(t)\mathbf{z}_j.
$$

---

# 11. News Revision Control

Articles may be edited after publication.

Therefore the final article body must not automatically be assigned to the original publication timestamp.

Ideally, the data model stores revisions:

$$
x_j^{(1)},
x_j^{(2)},
\ldots,
x_j^{(m)}.
$$

Each revision has its own availability time

$$
t_{j,r}^{available}.
$$

If historical revision information cannot be obtained, that limitation must be documented explicitly.

---

# 12. Blockchain Data

Candidate blockchain variables include:

- transaction count,
- transaction volume,
- active addresses,
- exchange inflows,
- exchange outflows,
- miner flows,
- fee levels,
- mempool state,
- hash rate,
- mining difficulty,
- large-holder activity.

A generic observation schema is:

| Field | Description |
|---|---|
| metric | variable name |
| observation_time | blockchain/event period |
| publish_time | provider availability |
| receive_time | acquisition |
| value | metric value |
| source | data provider |

---

# 13. Blockchain Confirmation Problem

Blockchain event time and knowledge time are not necessarily identical.

For transaction $j$,

$$
t_j^{broadcast}
\leq
t_j^{block}
\leq
t_j^{confirmed}
\leq
t_j^{provider}.
$$

The usable timestamp depends on the feature definition.

For example:

- mempool data may be observable before confirmation,
- confirmed transaction metrics cannot be backdated to broadcast time,
- provider aggregates may appear significantly later.

Therefore:

$$
t_j^{usable}
\neq
t_j^{economic}
$$

in many cases.

---

# 14. Mining State

Candidate mining variables include:

$$
HashRate_t,
$$

$$
Difficulty_t,
$$

$$
MinerFlow_t,
$$

$$
FeeRevenue_t.
$$

Energy-related variables may be added only where an economically plausible mechanism exists.

They will not be included merely because they correlate historically with Bitcoin.

---

# 15. Institutional and ETF Flow Data

Where reliable point-in-time data are available, candidate variables may include:

- ETF subscriptions,
- ETF redemptions,
- fund flows,
- futures positioning,
- institutional holdings.

The key timestamp is not necessarily the economic date of the flow.

The relevant timestamp is:

$$
t^{public}
=
\text{time the information became observable}.
$$

Daily values released after market hours cannot be used as if known during that trading day.

---

# 16. Cross-Market Data

Candidate cross-market variables include:

- Ethereum,
- major crypto assets,
- US equity indices,
- volatility indices,
- Treasury yields,
- interest-rate futures,
- US dollar indices,
- gold,
- oil,
- liquidity measures.

A generic schema is:

| Field | Description |
|---|---|
| timestamp | observation time |
| asset | identifier |
| venue | source |
| field | price / yield / volume etc. |
| value | observed value |
| availability_time | usable timestamp |

---

# 17. Market-Hours Problem

Bitcoin trades continuously.

Many traditional markets do not.

Therefore simply forward-filling traditional asset prices creates a potentially misleading representation.

For market $j$, define the market-open indicator

$$
M_{j,t}
=
\mathbf{1}
\{
\text{market }j\text{ is actively trading at }t
\}.
$$

Cross-market features may therefore include both

$$
X_{j,t}
$$

and

$$
M_{j,t}.
$$

The model can then distinguish a genuinely unchanged price from a closed market.

---

# 18. Stale-Price Age

For cross-market variable $X_j$, let

$$
t_j^{last}
$$

be the timestamp of its most recent observation.

Define observation age

$$
Age_{j,t}
=
t-t_j^{last}.
$$

This becomes a model feature.

A value observed one minute ago should not necessarily be treated identically to one observed fourteen hours ago.

---

# 19. Macroeconomic Data

Possible variables include:

- inflation,
- employment,
- policy rates,
- central-bank announcements,
- liquidity measures,
- economic-surprise indicators.

Macro data require three timestamps:

$$
t^{reference},
$$

the period the statistic describes,

$$
t^{release},
$$

the first public release,

and

$$
t^{revision}.
$$

Prediction features must use

$$
t^{release},
$$

not $t^{reference}$.

Future revisions must never overwrite the historical value used by the model.

---

# 20. Volatility and Derivatives Data

Candidate derivatives-state features include:

- implied volatility,
- term structure,
- skew,
- futures basis,
- funding rates,
- open interest,
- liquidation activity,
- options volume.

A derivatives-state vector may be represented as

$$
\mathbf{v}_t
=
[
IV_t,
Skew_t,
Basis_t,
Funding_t,
OI_t,
Liquidation_t
]^\top.
$$

The advanced-volatility literature may later support more sophisticated representations if economically justified.

---

# 21. Cross-Venue Information

Bitcoin price formation occurs across multiple exchanges.

For venue $j$, let

$$
m_t^{(j)}
$$

denote its midquote.

Relative dislocation between venues $i$ and $j$ is

$$
d_t^{(i,j)}
=
\log m_t^{(i)}
-
\log m_t^{(j)}.
$$

Potential research questions include whether:

$$
d_t^{(i,j)}
$$

predicts subsequent convergence,

and whether order flow on one venue leads price adjustment on another.

---

# 22. Clock Synchronization

Cross-source research requires a common time basis.

All timestamps should ultimately be converted to

$$
UTC.
$$

However, original timestamps should also be retained.

The raw layer should therefore store both:

```text
source_timestamp
utc_timestamp
