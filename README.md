# When Does Better Inference Make a Better Market Maker?

This project tests whether better inference about persistent order flow leads to better market-making decisions in a synthetic simulation. I hold a simple quoting rule fixed and compare how different beliefs about a latent order-flow regime affect forecasts, quotes, fills, inventory, and terminal P&L.

## Main result

Across 300 evaluation paths of 1,000 events, the HMM-informed policy earned a mean terminal P&L of **209.28** normalized units versus **196.44** for the inventory-only baseline, an average uplift of **12.83** with a 95% confidence interval of **[10.80, 14.86]**.

The more interesting result is that better forecasting didn't always lead to better trading decisions. Across 49 misspecified HMMs, Brier score and P&L uplift had a Spearman correlation of **−0.427** (lower Brier is better). The highest-P&L model in that grid was more persistent than the correct one and had a *worse* Brier score.

![Inference quality versus decision quality](figures/inference_vs_decision.png)

## Model

The simulator runs in event time with a three-state latent order-flow regime:

$$
Z_t \in \{-1,0,+1\}
$$

$$
P(Y_t=+1 \mid Z_t=z)=\frac{1}{2}+\beta z
$$

$$
S_{t+1}=S_t+\mu Z_t+\sigma\varepsilon_{t+1},
\qquad
\varepsilon_{t+1}\sim\mathcal{N}(0,1)
$$

Here, $Z_t$ is the latent regime, $Y_t$ is the public trade sign, and $S_t$ is the midprice. `MarketParams` defines the synthetic market dynamics and shared execution settings. `FilterParams` defines the transition matrix, emission strength, and economic map used by the Bayesian policy.

Each seed generates one immutable `MarketPath` containing latent states, public trade signs, Gaussian shocks, prices, and execution uniforms. Every policy is evaluated on the same path.

### Bayesian filter

Given the previous posterior, the one-step predictive belief is

$$
\bar{\pi}_t(j)
=
\sum_i \pi_{t-1}(i)P_{ij}
$$

Before quoting, the policy computes side-conditional posteriors for the two possible next trade signs:

$$
\pi_t^{(y)}(j)
\propto
P(Y_t=y \mid Z_t=j)\bar{\pi}_t(j),
\qquad
y\in\{-1,+1\}
$$

We use these beliefs to adjust the two quote sides for adverse selection:

$$
c_a
=
\hat{\mu}\,\mathbb{E}[Z_t \mid Y_t=+1]
$$

$$
c_b
=
-\hat{\mu}\,\mathbb{E}[Z_t \mid Y_t=-1]
$$

After the public sign arrives, the posterior is updated with the same Bayes rule. The filter is implemented directly in NumPy and uses the HMM parameters supplied by the synthetic model.

### Quoting rule

Execution at quote distance $\delta$ has probability

$$
p_{\mathrm{fill}}(\delta)
=
p_0 e^{-k\delta}
$$

With quadratic inventory penalty

$$
R(q)=\eta q^2,
$$

the side objectives are

$$
J_a(\delta)
=
p_0 e^{-k\delta}
\left[\delta-c_a+\eta(2q-1)\right]
$$

$$
J_b(\delta)
=
p_0 e^{-k\delta}
\left[\delta-c_b-\eta(2q+1)\right]
$$

This gives the interior quote distances

$$
\delta_a^{\ast}
=
\frac{1}{k}+c_a-\eta(2q-1)
$$

$$
\delta_b^{\ast}
=
\frac{1}{k}+c_b+\eta(2q+1)
$$

The implementation uses $\max(0,\delta^{\ast})$ as a minimum-distance safeguard. This gives a **myopic** quoting policy that depends on the current belief state.

## Policies

- **Fixed:** uses a constant symmetric distance $1/k$ and ignores inference and inventory.
- **Inventory:** uses the same quoting rule with adverse-selection costs set to zero.
- **Rolling:** fits a conditional OLS model of price changes using trailing sign imbalance, trade side, and their interaction on separate training paths. At quote time, it evaluates the fitted model once for a hypothetical buy and once for a hypothetical sell. $W\in\{5,10,20,50,100\}$ is selected by validation MSE.
- **Bayesian:** uses HMM side-conditional posteriors and the assumed $\mu$ map.
- **Oracle:** uses the latent state $Z_t$ directly as a full-information benchmark.

Cash and inventory update at the quoted execution price. Wealth is $X_t+q_tS_t$, with a common terminal liquidation charge of $0.10\lvert q_T\rvert$. Reported markout is the maker-side one-event loss $Y_t(S_{t+1}-S_t)$ averaged over fills.

## Reproduce

Python 3.11 or newer is required. From a clean checkout:

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
.venv/bin/ruff check .

.venv/bin/python experiments/benchmark.py --paths 300 --sweep-paths 80 --horizon 1000
.venv/bin/python experiments/misspecification.py --paths 80 --horizon 1000
.venv/bin/python experiments/inference_vs_decision.py
