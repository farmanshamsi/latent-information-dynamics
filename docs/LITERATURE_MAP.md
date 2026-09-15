# Literature Map

## Latent Information Dynamics and Endogenous Price Formation in Bitcoin

This document maps the research programme to its theoretical and empirical literature.

It is not intended to be the final literature review.

The purpose of this first version is to identify:

- the theoretical foundations of the project,
- the principal competing hypotheses,
- the empirical literature associated with each information channel,
- the methodological literature required for estimation,
- the gaps that the proposed research attempts to address.

The final paper-level literature review will expand this map toward approximately 40–60 directly relevant academic papers.

---

# 1. Central Research Question

The project asks whether information observable before a Bitcoin price adjustment contains predictive information beyond Bitcoin's own historical price.

The central comparison is

$$
\mathcal{M}_{P}
\quad\text{vs.}\quad
\mathcal{M}_{I}
\quad\text{vs.}\quad
\mathcal{M}_{I+P}.
$$

The relevant literature therefore spans several research traditions rather than a single forecasting literature.

The major literature blocks are:

1. market efficiency and information incorporation,
2. market microstructure and price discovery,
3. order flow and limit-order-book dynamics,
4. cross-market equilibrium and cointegration,
5. textual information and financial NLP,
6. cryptocurrency-specific market structure,
7. blockchain and crypto-economic information,
8. machine learning and multimodal information fusion,
9. latent states and regime dependence,
10. validation, leakage and backtest overfitting.

---

# 2. Market Efficiency and Information Incorporation

## 2.1 Efficient Markets as the Counter-Hypothesis

A necessary starting point is the efficient-markets literature.

Fama's classical formulation of informational efficiency provides an important null hypothesis for this project.

If observable public information is incorporated rapidly and sufficiently into Bitcoin price, then non-price variables should contribute little incremental predictive power once current and historical price information is conditioned upon.

The project therefore does not begin by assuming that information necessarily predicts price.

Instead, the efficient-markets interpretation provides a falsifiable competing explanation:

$$
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
\approx 0.
$$

### Anchor Reference

Fama, E. F. (1970).  
**Efficient Capital Markets: A Review of Theory and Empirical Work.**  
*Journal of Finance*, 25(2), 383–417.

---

## 2.2 Costly Information and Incomplete Price Revelation

Grossman and Stiglitz provide an important theoretical counterpoint to the strongest interpretation of informational efficiency.

If information acquisition is costly, markets cannot perfectly reveal all information while simultaneously preserving incentives to acquire that information.

This is relevant to the project because several proposed information channels require costly or technically sophisticated processing:

- high-frequency order-book reconstruction,
- blockchain-state analysis,
- cross-market monitoring,
- news extraction,
- machine-learning representation.

The project therefore investigates whether measurable information-processing frictions leave temporary predictive structure.

### Anchor Reference

Grossman, S. J., & Stiglitz, J. E. (1980).  
**On the Impossibility of Informationally Efficient Markets.**  
*American Economic Review*, 70(3), 393–408.

---

# 3. Market Microstructure and Information Transmission

The project treats price formation as a market mechanism rather than merely a statistical time series.

This makes market microstructure one of the central theoretical foundations.

---

## 3.1 Information-Based Trading

Kyle develops a canonical model in which informed traders strategically trade against noise traders while market makers infer information from order flow.

This creates the fundamental sequence

$$
\text{private information}
\rightarrow
\text{order flow}
\rightarrow
\text{market-maker inference}
\rightarrow
\text{price adjustment}.
$$

This mechanism is closely related to the project's central intuition.

Price need not adjust instantaneously to the information source itself.

Information can become progressively revealed through trading activity.

### Anchor Reference

Kyle, A. S. (1985).  
**Continuous Auctions and Insider Trading.**  
*Econometrica*, 53(6).

---

## 3.2 Empirical Market Microstructure

Hasbrouck provides the empirical framework linking:

- trades,
- quotes,
- price impact,
- information asymmetry,
- liquidity,
- market depth,
- price discovery.

This literature motivates the distinction between transaction prices and the evolving informational state of the market.

### Core Source

Hasbrouck, J.  
**Empirical Market Microstructure.**

This source will guide the project's treatment of:

$$
\text{trade innovation}
\rightarrow
\text{quote response}
\rightarrow
\text{price discovery}.
$$

---

# 4. Order Flow and Limit-Order-Book Information

## 4.1 Order-Flow Imbalance

Cont, Kukanov and Stoikov provide an important empirical foundation for the project's microstructure layer.

Their work studies:

- market orders,
- limit orders,
- cancellations,
- order-flow imbalance,
- market depth,
- short-horizon price changes.

The important implication for this research is that volume alone is not necessarily the relevant state variable.

The imbalance between incoming supply and demand can be more informative.

This directly motivates features such as

$$
OFI_t,
$$

$$
OBI_t,
$$

$$
TI_t,
$$

and depth-conditioned price impact.

### Anchor Reference

Cont, R., Kukanov, A., & Stoikov, S. (2014).  
**The Price Impact of Order Book Events.**  
*Journal of Financial Econometrics*, 12(1), 47–88.

---

## 4.2 Research Implication

The microstructure hypothesis can be expressed as

$$
I
\left(
Y_{t,h};
\mathbf{o}_t
\mid
\mathbf{p}_t
\right)
>0.
$$

The question is not simply whether order-book imbalance correlates with contemporaneous return.

The stronger research question is whether the state of liquidity and order flow at time $t$ contains information about price adjustment over $t+h$.

---

# 5. Cryptocurrency Market Structure

Bitcoin differs from a centralized equity market because price discovery is distributed across multiple venues.

This creates an important additional research dimension.

---

## 5.1 Cross-Exchange Fragmentation

Makarov and Schoar document economically meaningful price differences and arbitrage opportunities across cryptocurrency exchanges.

They also study signed volume and common versus exchange-specific trading components.

This supports investigation of:

- cross-exchange price differences,
- venue-specific order flow,
- fragmented liquidity,
- lead-lag relationships,
- common versus idiosyncratic flow.

### Anchor Reference

Makarov, I., & Schoar, A. (2020).  
**Trading and Arbitrage in Cryptocurrency Markets.**  
*Journal of Financial Economics*, 135(2), 293–319.

---

## 5.2 Crypto-Specific Return Predictors

Liu and Tsyvinski examine whether cryptocurrency returns behave like traditional financial assets and investigate cryptocurrency-specific predictors.

Their work motivates examination of variables related to:

- network activity,
- investor attention,
- crypto-specific momentum,
- adoption.

### Anchor Reference

Liu, Y., & Tsyvinski, A. (2021).  
**Risks and Returns of Cryptocurrency.**  
*Review of Financial Studies*, 34(6), 2689–2727.

---

# 6. Cross-Market Equilibrium and Disequilibrium

The project does not treat correlation as evidence of predictive structure.

Cross-market relationships must be examined through formal time-series methods.

---

## 6.1 Cointegration

Engle and Granger establish the connection between non-stationary time series, cointegration and error-correction dynamics.

If two integrated processes satisfy

$$
y_t-\beta x_t
\sim I(0),
$$

then the stationary residual represents deviation from a long-run equilibrium relationship.

This motivates the research variable

$$
z_t
=
y_t-\alpha-\beta x_t.
$$

The relevant question is whether $z_t$ predicts subsequent Bitcoin adjustment.

### Anchor Reference

Engle, R. F., & Granger, C. W. J. (1987).  
**Co-integration and Error Correction: Representation, Estimation, and Testing.**  
*Econometrica*, 55(2), 251–276.

---

## 6.2 Multivariate Cointegration

Johansen extends cointegration analysis to multivariate VAR systems.

This is relevant because Bitcoin price formation may involve more than a pairwise relationship.

Potential systems may contain:

$$
BTC,
\quad
ETH,
\quad
USD,
\quad
rates,
\quad
equities,
\quad
gold,
\quad
liquidity.
$$

The corresponding VECM representation is

$$
\Delta\mathbf{Z}_t
=
\alpha\beta^\top\mathbf{Z}_{t-1}
+
\sum_i
\Gamma_i
\Delta\mathbf{Z}_{t-i}
+
\boldsymbol{\varepsilon}_t.
$$

### Anchor Reference

Johansen, S. (1991).  
**Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models.**  
*Econometrica*, 59(6), 1551–1580.

---

## 6.3 Econometric Foundation

Enders provides the broader time-series framework required for:

- stationarity,
- unit-root testing,
- VAR models,
- impulse responses,
- Granger-style predictive relationships,
- cointegration,
- error correction.

### Core Source

Enders, W.  
**Applied Econometric Time Series.**

This source forms the principal econometric reference for the cross-market layer.

---

# 7. Financial Text and Machine-Readable News

News is not useful merely because text exists.

It must be transformed into a measurable point-in-time information state.

---

## 7.1 Financial Language Models

FinBERT demonstrates the value of adapting pretrained language models to domain-specific financial language.

This supports the use of specialized models rather than generic sentiment lexicons.

### Anchor Reference

Araci, D. (2019).  
**FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.**

---

## 7.2 Beyond Sentiment

The intended news representation is broader than positive-versus-negative sentiment.

The project will investigate structured variables such as

$$
\text{event type},
$$

$$
\text{entity},
$$

$$
\text{novelty},
$$

$$
\text{uncertainty},
$$

$$
\text{relevance},
$$

$$
\text{directional implication}.
$$

The proposed chain is

$$
\text{text}
\rightarrow
\text{structured event representation}
\rightarrow
\text{information state}
\rightarrow
\text{forecast}.
$$

LLMs are therefore initially treated as information-extraction systems rather than autonomous trading systems.

---

# 8. Blockchain and Crypto-Economic Information

Bitcoin offers a public transaction ledger unavailable in traditional financial markets.

This creates a potentially distinct information layer.

Candidate variables include:

$$
\text{exchange flows},
$$

$$
\text{miner flows},
$$

$$
\text{network activity},
$$

$$
\text{hash rate},
$$

$$
\text{difficulty},
$$

$$
\text{stablecoin liquidity},
$$

$$
\text{large-holder activity}.
$$

The literature review must distinguish carefully between:

1. variables representing economic mechanisms,
2. variables that merely correlate retrospectively with price.

The core hypothesis is

$$
I
\left(
Y_{t,h};
\mathbf{c}_t
\mid
\mathbf{p}_t
\right)
>0.
$$

This literature block requires significant expansion during the dedicated external literature-search phase.

---

# 9. Information Fusion and Machine Learning

The project contains heterogeneous modalities.

A single model need not be appropriate for every information source.

The machine-learning literature therefore enters mainly through:

- specialist models,
- representation learning,
- feature selection,
- ensemble learning,
- probability fusion.

---

## 9.1 Specialist Models

Each domain can first produce its own representation:

$$
\mathbf{z}_t^{O}
=
f_O(\mathbf{o}_t),
$$

$$
\mathbf{z}_t^{N}
=
f_N(\mathbf{n}_t),
$$

$$
\mathbf{z}_t^{C}
=
f_C(\mathbf{c}_t),
$$

$$
\mathbf{z}_t^{X}
=
f_X(\mathbf{x}_t).
$$

These can then be combined through

$$
\mathbf{z}_t
=
g
\left(
\mathbf{z}_t^{O},
\mathbf{z}_t^{N},
\mathbf{z}_t^{C},
\mathbf{z}_t^{X}
\right).
$$

---

## 9.2 Ensemble Learning

The CQF ensemble material motivates comparison between:

- voting,
- weighted voting,
- stacking,
- blending,
- heterogeneous model ensembles.

The important research principle is that ensemble construction must itself remain out of sample.

Meta-models must never be trained on predictions generated from data already seen by the base estimator.

---

## 9.3 Machine Learning in Finance

Dixon, Halperin and Bilokon provide a broad framework for applying machine learning to financial problems including:

- high-dimensional data,
- neural networks,
- hidden states,
- transaction networks,
- order-book prediction,
- reinforcement learning.

### Core Source

Dixon, M., Halperin, I., & Bilokon, P.  
**Machine Learning in Finance: From Theory to Practice.**

---

# 10. Latent States and Regime Dependence

The project's title refers to a latent information state.

This does not imply that a latent-variable model must necessarily be used.

Instead, the hypothesis is that several observable information channels may reflect a lower-dimensional underlying market state.

Possible methods include:

$$
\text{PCA},
$$

$$
\text{state-space models},
$$

$$
\text{hidden Markov models},
$$

$$
\text{autoencoders},
$$

$$
\text{representation learning}.
$$

The research question is whether

$$
\mathbf{i}_t
\rightarrow
\mathbf{z}_t
$$

produces a representation that is more stable or informative than raw feature concatenation.

Regime dependence also permits

$$
P(Y_{t,h}=1\mid\mathbf{i}_t,S_t=k)
$$

to differ across market environments.

---

# 11. Volatility and Stochastic-State Literature

Volatility is potentially useful as a latent market-state variable but is not the principal object of the research.

The advanced-volatility material provides mathematical foundations for:

- local volatility,
- stochastic volatility,
- Heston dynamics,
- volatility surfaces,
- latent variance,
- Fourier methods,
- stochastic differential equations.

This literature may become relevant if volatility state materially modifies information transmission.

It should not be introduced merely to increase mathematical complexity.

---

# 12. Long Memory and Fractional Processes

Fractional Brownian motion provides a possible framework for studying persistent dependence.

For Hurst parameter $H$,

$$
H>\frac{1}{2}
$$

is associated with persistent dependence in fractional Gaussian noise, while

$$
H<\frac{1}{2}
$$

corresponds to anti-persistence.

However, fractional Brownian motion introduces important theoretical difficulties because it is not generally a semimartingale for

$$
H\neq\frac{1}{2}.
$$

Therefore fractional models are currently classified as an optional research extension rather than part of the core architecture.

---

# 13. Validation and Financial Machine Learning

A sophisticated model can produce a sophisticated-looking false result.

Validation methodology is therefore part of the scientific contribution.

---

## 13.1 Backtest Overfitting

Bailey, Borwein, López de Prado and Zhu demonstrate how repeated search over strategy configurations can generate impressive in-sample backtests even without genuine economic signal.

This directly motivates restrictions on:

- repeated parameter search,
- repeated feature search,
- model selection on test data,
- unreported specification search.

### Anchor Reference

Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014).  
**Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.**

---

## 13.2 Financial ML Validation

López de Prado provides methods relevant to:

- purged validation,
- embargoes,
- financial labeling,
- feature importance,
- structural breaks,
- backtest overfitting,
- probabilistic performance measures.

### Core Source

López de Prado, M.  
**Advances in Financial Machine Learning.**

These methods will inform the project's final validation protocol.

---

# 14. Literature-to-Hypothesis Mapping

## H1 — Incremental non-price information

$$
I
\left(
Y_{t,h};
\mathbf{i}_t
\mid
\mathbf{p}_t
\right)
>0
$$

Relevant literature:

- Fama,
- Grossman–Stiglitz,
- Kyle,
- Hasbrouck.

---

## H2 — Microstructure contains short-horizon predictive information

$$
I
\left(
Y_{t,h};
\mathbf{o}_t
\mid
\mathbf{p}_t
\right)
>0
$$

Relevant literature:

- Kyle,
- Hasbrouck,
- Cont–Kukanov–Stoikov,
- cryptocurrency microstructure research.

---

## H3 — Cross-market disequilibrium predicts adjustment

$$
P
\left(
Y_{t,h}
\mid
z_t
\right)
\neq
P(Y_{t,h})
$$

Relevant literature:

- Engle–Granger,
- Johansen,
- Enders,
- Makarov–Schoar.

---

## H4 — Text contains incremental information

$$
I
\left(
Y_{t,h};
\mathbf{n}_t
\mid
\mathbf{p}_t
\right)
>0
$$

Relevant literature:

- FinBERT,
- financial NLP,
- event extraction,
- financial transformer models.

---

## H5 — Crypto-economic state contains incremental information

$$
I
\left(
Y_{t,h};
\mathbf{c}_t
\mid
\mathbf{p}_t
\right)
>0
$$

Relevant literature:

- Liu–Tsyvinski,
- blockchain analytics,
- exchange-flow research,
- miner-flow research,
- network-adoption research.

---

## H6 — Information contribution changes across regimes

$$
P
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,S_t=k
\right)
\neq
P
\left(
Y_{t,h}=1
\mid
\mathbf{i}_t,S_t=l
\right).
$$

Relevant literature:

- hidden-state models,
- regime-switching models,
- volatility-state models,
- latent representation learning.

---

# 15. Preliminary Research Gap

The existing literature contains strong individual research streams on:

- efficient markets,
- information-based trading,
- order-flow price impact,
- cryptocurrency fragmentation,
- cryptocurrency-specific predictors,
- financial NLP,
- cointegration,
- machine learning.

The proposed project is positioned at their intersection.

The intended contribution is not to claim that each source individually predicts Bitcoin.

Instead, the study asks whether multiple point-in-time information channels can be organized into a common information-state framework and evaluated against a deliberately strong price-only benchmark.

The central empirical contribution is therefore the comparison

$$
\mathcal{M}_{P}
\quad\text{vs.}\quad
\mathcal{M}_{I}
\quad\text{vs.}\quad
\mathcal{M}_{I+P},
$$

combined with domain-level ablation and strict point-in-time validation.

---

# 16. Literature Expansion Plan

The next literature-search phase will expand this architecture into a paper-level review.

Target literature groups are:

| Research block | Approximate target |
|---|---:|
| Information efficiency / price discovery | 5–7 papers |
| Market microstructure / order flow | 8–10 papers |
| Bitcoin / crypto microstructure | 6–8 papers |
| News / NLP / information events | 6–8 papers |
| On-chain / network information | 5–7 papers |
| Cross-market / cointegration | 5–7 papers |
| Latent-state / multimodal ML | 5–7 papers |
| Validation / financial ML | 4–6 papers |

Some papers may contribute to more than one block.

The final objective is not a fixed numerical paper count.

The objective is sufficient literature coverage to justify every major assumption, feature family, model class and validation decision in the empirical study.

---

# 17. Current Literature Position

At the present stage, the literature suggests three competing possibilities.

### Information-efficiency interpretation

Observable information is incorporated rapidly enough that additional information states provide little incremental predictive value.

### Partial-information-adjustment interpretation

Different information sources reach different market participants at different speeds, allowing temporary predictive structure before complete price adjustment.

### State-dependent interpretation

Information is incorporated efficiently in some regimes but more slowly in others because liquidity, attention, volatility and market structure vary over time.

The empirical research will distinguish among these explanations rather than assuming one in advance.
