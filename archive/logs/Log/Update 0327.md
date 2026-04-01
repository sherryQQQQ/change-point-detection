![[Pasted image 20260327231513.png]]![[Pasted image 20260327231532.png]]
Window size problem fixed.
Future information problem fixed :
		running_std_max[i] = max(rolling_std[i], running_std_max[i-1] if i > 0 else 0.0)
		 ws = ws_min + (ws_max - ws_min) *(1-rolling_std / running_std_max)

*try another method exponential moving average if not smooth

**Performance issue:**
Even with narrower window -> adaptive window has worse performance.
Change ws_min from 2 to 3:

![[Pasted image 20260327231836.png]]

sensitivity to change point

---

## Bayesian Online Change Point Detection (BOCPD)

### Background

BOCPD (Adams & MacKay 2007) detects regime shifts in streaming data by maintaining a posterior distribution over the **run length** $r_t$ (how many steps since the last changepoint). At each new observation $x_t$, it updates this distribution and outputs a changepoint probability $P(r_t = 0)$.

We use BOCPD as the change-detection layer in our predictor: when $P(r_t=0)$ exceeds a threshold, the OLS model refits on the new regime.

### Formulas

**Prior:** Normal-Inverse-Gamma conjugate on observation mean and variance.

$$\mu \sim \mathcal{N}(\mu_0,\; (\kappa_0 \sigma^2)^{-1}), \quad \sigma^2 \sim \text{Inv-Gamma}(\alpha_0,\; \beta_0)$$

**Predictive distribution** (Student-t) for observation $x_t$ given run length $r$:

$$x_t \mid r \;\sim\; t_{2\alpha_r}\!\left(\mu_r,\; \frac{\beta_r(\kappa_r+1)}{\alpha_r \kappa_r}\right)$$

**Posterior update** (sufficient statistics after seeing $x_t$):

$$\kappa_{r+1} = \kappa_r + 1, \quad \mu_{r+1} = \frac{\kappa_r \mu_r + x_t}{\kappa_{r+1}}, \quad \alpha_{r+1} = \alpha_r + \tfrac{1}{2}, \quad \beta_{r+1} = \beta_r + \frac{\kappa_r(x_t - \mu_r)^2}{2\kappa_{r+1}}$$

**Run-length recursion:**

$$P(r_t=0) = \sum_r P(r_{t-1}=r)\;\pi(x_t \mid r)\;H(r) \quad \text{(changepoint)}$$
$$P(r_t=r+1) = P(r_{t-1}=r)\;\pi(x_t \mid r)\;(1-H(r)) \quad \text{(growth)}$$

where $H(r) = \frac{1}{\lambda}\left(1 + \frac{r}{\lambda}\right)$ is the hazard function (increasing with run length, capped at 0.5).

**Parameters in our implementation:**

| Parameter         | Symbol     | Role                                                        |
| ----------------- | ---------- | ----------------------------------------------------------- |
| `hazard_lambda`   | $\lambda$  | Expected run length between changepoints                    |
| `mu0`             | $\mu_0$    | Prior mean (set to long-run mean $b$)                       |
| `kappa0`          | $\kappa_0$ | Prior strength on mean (pseudo-observations)                |
| `alpha0`          | $\alpha_0$ | Prior shape for variance (need $>1$ for proper prior)       |
| `beta0`           | $\beta_0$  | Prior rate for variance; prior var $= \beta_0/(\alpha_0-1)$ |
| `alarm_threshold` | -          | Trigger refit when $P(r_t=0) >$ threshold                   |

### Results

**Adaptive window fixes applied:**
- Window size: causal rolling-std / running-std-max schedule (no future information)
  ```
  running_std_max[i] = max(rolling_std[i], running_std_max[i-1])
  ws = ws_min + (ws_max - ws_min) * (1 - rolling_std / running_std_max)
  ```
- `alarm_threshold` bug fixed (was hardcoded 0.01, now uses configured value)
- BOCPD prior aligned to data: `mu0=80`, `alpha0=2`, `beta0=5`

**Performance issue:** adaptive window still underperforms fixed window in some configs (change `ws_min` from 2 to 3 helps).

Sensitivity to changepoint detection threshold remains a tuning concern. 
/Users/qianxinhui/Desktop/Misc/2026/NU-Research/kellogg/change-point-detection/Log/adaptive_window_grid_search_a03_b80.pdf
versu.
/Users/qianxinhui/Desktop/Misc/2026/NU-Research/kellogg/change-point-detection/Log/for_histogram_CoxM1_Z05_serv10_t1.pdf

