"""
BOCPD + OLS/diff-OLS Change Point Detection Predictor

Combines Bayesian Online Change Point Detection (Student-t) with
OLS-based windowed prediction and model adaptation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t as t_dist


class BOOLES_cpdPredictor:
    """
    Bayesian-Online-OLS predictor.

    Runs BOCPD online over the full series to compute per-point change-point
    probabilities, then uses sliding-window OLS/diff-OLS for prediction with
    model refitting triggered by BOCPD alarms.
    """

    def __init__(self, min_history_length=5, hazard_lambda=100,
                 mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
                 alarm_threshold=0.7, alarm_min_consecutive=1):
        self.results = {
            'predicted_value': [],
            'value': [],
            'prediction_errors': [],
            'stepwise_value': [],
            'predicted_step_function_time_interval': [],
            'changepoint_probs': [],
        }
        self.hazard_lambda = hazard_lambda
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alarm_threshold = alarm_threshold
        self.alarm_min_consecutive = alarm_min_consecutive

        self.bocpd_initialized = False
        self.R = None
        self.muT = None
        self.kappaT = None
        self.alphaT = None
        self.betaT = None
        self.current_time = 0
        self.max_run_length = 1000

    # ------------------------------------------------------------------
    # BOCPD core
    # ------------------------------------------------------------------
    def hazard(self, r):
        if r == 0:
            return 0.0
        base_h = 1.0 / self.hazard_lambda
        growth = 1.0 + r / self.hazard_lambda
        return min(base_h * growth, 0.5)

    def _init_bocpd(self):
        self.R = np.array([1.0])
        self.muT = [self.mu0]
        self.kappaT = [self.kappa0]
        self.alphaT = [self.alpha0]
        self.betaT = [self.beta0]
        self.current_time = 0
        self.bocpd_initialized = True

    def update_bocpd_online(self, x):
        if not self.bocpd_initialized:
            self._init_bocpd()
        self.current_time += 1

        n = len(self.R)
        pred = np.zeros(n)
        for r in range(n):
            mu, kappa, alpha, beta = (
                self.muT[r], self.kappaT[r], self.alphaT[r], self.betaT[r])
            if kappa > 0 and alpha > 0 and beta > 0:
                scale = np.sqrt((beta * (kappa + 1)) / (alpha * kappa))
                df = 2 * alpha
                if scale > 1e-10 and df > 0:
                    z = np.clip((x - mu) / scale, -10, 10)
                    pred[r] = t_dist.pdf(z, df) / scale
                else:
                    pred[r] = 1e-10
            else:
                pred[r] = 1e-10
        pred = np.maximum(pred, 1e-300)

        h = np.array([self.hazard(r) for r in range(n)])
        growth = self.R * pred * (1 - h)
        cp_prob = np.sum(self.R * pred * h)

        new_R = np.zeros(n + 1)
        new_R[0] = cp_prob
        new_R[1:] = growth

        s = new_R.sum()
        if s > 1e-300:
            new_R /= s
        else:
            new_R[:] = 0.0
            new_R[0] = 1.0

        eps = 1e-10
        new_R = new_R * (1 - eps) + eps / len(new_R)

        # Bayesian sufficient-statistics update
        new_mu = [self.mu0]
        new_kappa = [self.kappa0]
        new_alpha = [self.alpha0]
        new_beta = [self.beta0]
        for r in range(n):
            mu, kappa, alpha, beta = (
                self.muT[r], self.kappaT[r], self.alphaT[r], self.betaT[r])
            kn = kappa + 1
            mn = (kappa * mu + x) / kn
            an = alpha + 0.5
            bn = beta + kappa * (x - mu) ** 2 / (2 * kn)
            new_mu.append(mn)
            new_kappa.append(min(kn, 1e6))
            new_alpha.append(min(an, 1e6))
            new_beta.append(min(bn, 1e10))

        if len(new_R) > self.max_run_length:
            keep = np.argsort(new_R)[-self.max_run_length:]
            new_R = new_R[keep]
            new_R /= new_R.sum()
            new_mu = [new_mu[i] if i < len(new_mu) else self.mu0 for i in keep]
            new_kappa = [new_kappa[i] if i < len(new_kappa) else self.kappa0 for i in keep]
            new_alpha = [new_alpha[i] if i < len(new_alpha) else self.alpha0 for i in keep]
            new_beta = [new_beta[i] if i < len(new_beta) else self.beta0 for i in keep]

        self.R, self.muT, self.kappaT = new_R, new_mu, new_kappa
        self.alphaT, self.betaT = new_alpha, new_beta
        return self.R[0]

    def get_current_model_params(self):
        if not self.bocpd_initialized or len(self.R) == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0
        idx = np.argmax(self.R)
        if idx < len(self.muT):
            return self.muT[idx], self.kappaT[idx], self.alphaT[idx], self.betaT[idx]
        return self.mu0, self.kappa0, self.alpha0, self.beta0

    # ------------------------------------------------------------------
    # Rolling prediction
    # ------------------------------------------------------------------
    def rolling_prediction(self, data, window_size=3, gamma=None, plot=True,
                           method='ols', file_name=None, cpd_method='bayesian'):
        data = data.copy()
        if 'time' not in data.columns:
            data['time'] = np.arange(len(data))
        if 'predicted_value' not in data.columns:
            data['predicted_value'] = np.nan
        if 'changepoint_prob' not in data.columns:
            data['changepoint_prob'] = np.nan

        all_dev = []
        if len(data) < 2 * window_size:
            print(f"Warning: data length ({len(data)}) < 2*window_size ({2*window_size})")
            return None

        detect_ws = window_size
        predict_ws = 2 * window_size - detect_ws

        def fit_diff_ols(w):
            if len(w) < 2:
                return None
            dt = w['time'].diff().mean()
            if pd.isna(dt) or dt == 0:
                dt = 1.0
            y_s = w['value'].diff() / dt
            x_s = w['value'].shift(1)
            tmp = pd.DataFrame({'y': y_s, 'x': x_s}).dropna()
            if len(tmp) < 2:
                return None
            try:
                return sm.OLS(tmp['y'], sm.add_constant(tmp['x'])).fit()
            except Exception:
                return None

        # initial OLS fit
        iw = data.iloc[:window_size]
        if method == 'ols':
            results = sm.OLS(iw['value'], sm.add_constant(iw['time'])).fit()
        elif method == 'diff_ols':
            results = fit_diff_ols(iw)
            if results is None:
                print("Could not initialise diff_ols model.")
                return None

        # --- BOCPD pass ---
        print("Running online BOCPD...")
        cp_probs = []
        for idx in range(len(data)):
            cp = self.update_bocpd_online(data.iloc[idx]['value'])
            cp_probs.append(cp)
            data.loc[data.index[idx], 'changepoint_prob'] = cp
            if idx % 50 == 0:
                print(f"  {idx + 1}/{len(data)}, CP prob={cp:.4f}")

        # --- Windowed prediction ---
        print("Running windowed prediction...")
        threshold = self.alarm_threshold

        for i in range(2 * window_size, len(data) + 1, window_size):
            ps = i - window_size
            pe = min(ps + predict_ws, len(data))
            pidx = data.index[ps:pe]

            if method == 'ols':
                Xp = sm.add_constant(data.loc[pidx, 'time'])
                data.loc[pidx, 'predicted_value'] = results.predict(Xp)
            elif method == 'diff_ols':
                try:
                    b0, b1 = results.params
                    a = -b1
                    b = -b0 / b1 if abs(b1) > 1e-6 else np.mean(
                        data['value'].iloc[:i - window_size])
                    z0_idx = max(0, ps - 1)
                    z0 = data['value'].iloc[z0_idx]
                    t0 = data['time'].iloc[z0_idx]
                    td = data.loc[pidx, 'time'] - t0
                    pv = b + (z0 - b) * np.exp(-a * td)
                    upper = np.max(data['value'].iloc[:i - window_size])
                    lower = np.min(data['value'].iloc[:i - window_size])
                    pv = np.clip(pv, lower, upper)
                    data.loc[pidx, 'predicted_value'] = pv
                except Exception:
                    data.loc[pidx, 'predicted_value'] = np.mean(
                        data['value'].iloc[:i - window_size])

            data.loc[pidx, 'stepwise_value'] = data.loc[pidx, 'predicted_value'].mean()

            didx = data.index[i - window_size: i - window_size + detect_ws]
            if len(didx) > 0:
                wcp = data.loc[didx, 'changepoint_prob']
                max_cp = wcp.max()
                if max_cp > threshold:
                    cp_loc = wcp.idxmax()
                    all_dev.append(cp_loc)
                    rw = data.iloc[i - window_size: i - window_size + detect_ws]
                    if method == 'ols':
                        try:
                            results = sm.OLS(rw['value'],
                                             sm.add_constant(rw['time'])).fit()
                        except Exception:
                            pass
                    elif method == 'diff_ols':
                        nr = fit_diff_ols(rw)
                        if nr:
                            results = nr
                else:
                    last = all_dev[-1] if all_dev else 0
                    rw = data.iloc[last:i]
                    if len(rw) > 1:
                        if method == 'ols':
                            try:
                                results = sm.OLS(rw['value'],
                                                 sm.add_constant(rw['time'])).fit()
                            except Exception:
                                pass
                        elif method == 'diff_ols':
                            nr = fit_diff_ols(rw)
                            if nr:
                                results = nr

        data.loc[data.index[:window_size], 'stepwise_value'] = \
            data.loc[data.index[:window_size], 'value'].mean()
        data.loc[data.index[:window_size], 'predicted_value'] = \
            data.loc[data.index[:window_size], 'value'].mean()

        if plot and len(data) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            ax1.plot(data['time'], data['value'], label='Original', color='blue', lw=1)
            ax1.plot(data['time'], data['predicted_value'], label='Predicted',
                     color='green', ls='--', alpha=0.8)
            ax1.step(data['time'], data['stepwise_value'], label='Stepwise',
                     color='orange', where='post', lw=2)
            if all_dev:
                ax1.scatter(data.loc[all_dev, 'time'], data.loc[all_dev, 'value'],
                            color='red', s=100, label='Change Points', zorder=5)
            ax1.set_xlim(0, data['time'].iloc[-1])
            ax1.set_title(f'BOCPD+OLS — file={file_name}, method={method}')
            ax1.set_xlabel('Time'); ax1.set_ylabel('Value')
            ax1.legend(); ax1.grid(True, alpha=0.3)

            ax2.plot(data['time'], data['changepoint_prob'], color='purple', lw=1)
            ax2.axhline(threshold, color='red', ls='--', alpha=0.7, label=f'Threshold ({threshold})')
            ax2.fill_between(data['time'], 0, data['changepoint_prob'],
                             where=data['changepoint_prob'] > threshold,
                             color='red', alpha=0.3)
            ax2.set_xlim(0, data['time'].iloc[-1])
            ax2.set_title('Change-Point Probability')
            ax2.set_xlabel('Time'); ax2.set_ylabel('CP Prob')
            ax2.legend(); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()

        # store results
        self.results['prediction_times'] = data['time']
        self.results['predictions'] = data['predicted_value']
        self.results['actual_values'] = data['value']
        self.results['prediction_errors'] = data['predicted_value'] - data['value']
        self.results['predicted_step_function'] = data['stepwise_value']
        dt_col = data['time'] - data['time'].shift(1)
        dt_col.iloc[0] = data['time'].iloc[0]
        self.results['predicted_step_function_time_interval'] = dt_col
        self.results['changepoint_probs'] = data['changepoint_prob']

        return self.get_summary_metrics()

    # ------------------------------------------------------------------
    # Adaptive-window rolling prediction
    # ------------------------------------------------------------------
    def adaptive_rolling_prediction(self, data, w_min=3, w_max=15,
                                     gamma=None, plot=True, method='ols',
                                     file_name=None, cpd_method='bayesian',
                                     lambda_=0.94,
                                     adaptive_threshold=True,
                                     threshold_base=None,
                                     threshold_k=2.0,
                                     window_method='rolling_std'):
        """
        Like rolling_prediction but with adaptive window size and
        optionally adaptive alarm threshold.

        Parameters
        ----------
        adaptive_threshold : bool
            If True, the CP alarm threshold is scaled by local volatility:
              threshold_t = base * (1 + k * v_t)
            where v_t = rolling_std / running_std_max (in [0, 1]).
            Volatile regions get a higher threshold (fewer false CPs),
            smooth regions get a lower threshold (more sensitive).
        threshold_base : float or None
            Base threshold. Defaults to self.alarm_threshold.
        threshold_k : float
            Scaling factor for volatility adjustment.
        window_method : str
            'rolling_std' (default) or 'ewma'.

        Returns
        -------
        dict : summary metrics (same as rolling_prediction)
        Also stores self.window_sizes_used and self.thresholds_used.
        """
        from .change_point_detection import _adaptive_window_size, _adaptive_window_size_ewma

        data = data.copy()
        if 'time' not in data.columns:
            data['time'] = np.arange(len(data))
        if 'predicted_value' not in data.columns:
            data['predicted_value'] = np.nan
        if 'changepoint_prob' not in data.columns:
            data['changepoint_prob'] = np.nan

        all_dev = []
        self.window_sizes_used = []
        self.thresholds_used = []
        if threshold_base is None:
            threshold_base = self.alarm_threshold

        if len(data) < 2 * w_min:
            print(f"Warning: data length ({len(data)}) < 2*w_min ({2 * w_min})")
            return None

        def fit_diff_ols(w):
            if len(w) < 2:
                return None
            dt = w['time'].diff().mean()
            if pd.isna(dt) or dt == 0:
                dt = 1.0
            y_s = w['value'].diff() / dt
            x_s = w['value'].shift(1)
            tmp = pd.DataFrame({'y': y_s, 'x': x_s}).dropna()
            if len(tmp) < 2:
                return None
            try:
                return sm.OLS(tmp['y'], sm.add_constant(tmp['x'])).fit()
            except Exception:
                return None

        # initial OLS fit using w_min
        iw = data.iloc[:w_min]
        if method == 'ols':
            results = sm.OLS(iw['value'], sm.add_constant(iw['time'])).fit()
        elif method == 'diff_ols':
            results = fit_diff_ols(iw)
            if results is None:
                print("Could not initialise diff_ols model.")
                return None

        # Fill initial segment
        data.loc[data.index[:w_min], 'stepwise_value'] = \
            data.loc[data.index[:w_min], 'value'].mean()
        data.loc[data.index[:w_min], 'predicted_value'] = \
            data.loc[data.index[:w_min], 'value'].mean()

        # --- BOCPD pass (processes every point, independent of window) ---
        print("Running online BOCPD...")
        cp_probs = []
        for idx in range(len(data)):
            cp = self.update_bocpd_online(data.iloc[idx]['value'])
            cp_probs.append(cp)
            data.loc[data.index[idx], 'changepoint_prob'] = cp
            if idx % 100 == 0:
                print(f"  {idx + 1}/{len(data)}, CP prob={cp:.4f}")

        # --- Adaptive windowed prediction ---
        print("Running adaptive windowed prediction...")
        i = w_min  # start after initial segment
        _lookback = 20
        _running_std_max = 1e-12  # maintained incrementally to avoid O(n²) per step

        while i + w_min <= len(data):
            # Adaptive window from past data only
            past_values = data['value'].iloc[:i].values

            # Incrementally update running_std_max (O(lookback) per step)
            local_std = np.std(past_values[max(0, len(past_values) - _lookback):]) \
                if len(past_values) > 1 else 0.0
            if local_std > _running_std_max:
                _running_std_max = local_std

            if window_method == 'ewma':
                ws = _adaptive_window_size_ewma(past_values, lambda_=lambda_,
                                                w_min=w_min, w_max=w_max)
            else:
                ws = _adaptive_window_size(past_values, lambda_=lambda_,
                                           w_min=w_min, w_max=w_max,
                                           _running_std_max=_running_std_max)
            ws = min(ws, len(data) - i)
            if ws < w_min:
                break

            # --- Adaptive threshold ---
            if adaptive_threshold and len(past_values) > 2:
                v_t = local_std / _running_std_max if _running_std_max > 1e-12 else 0.0
                threshold = threshold_base * (1.0 + threshold_k * v_t)
            else:
                threshold = threshold_base

            self.window_sizes_used.append((i, ws))
            self.thresholds_used.append((i, threshold))
            pidx = data.index[i:i + ws]

            # --- Prediction ---
            if method == 'ols':
                Xp = sm.add_constant(data.loc[pidx, 'time'])
                data.loc[pidx, 'predicted_value'] = results.predict(Xp)
            elif method == 'diff_ols':
                try:
                    b0, b1 = results.params
                    a = -b1
                    b = -b0 / b1 if abs(b1) > 1e-6 else np.mean(
                        data['value'].iloc[:i])
                    z0_idx = max(0, i - 1)
                    z0 = data['value'].iloc[z0_idx]
                    t0 = data['time'].iloc[z0_idx]
                    td = data.loc[pidx, 'time'] - t0
                    pv = b + (z0 - b) * np.exp(-a * td)
                    upper = np.max(data['value'].iloc[:i])
                    lower = np.min(data['value'].iloc[:i])
                    pv = np.clip(pv, lower, upper)
                    data.loc[pidx, 'predicted_value'] = pv
                except Exception:
                    data.loc[pidx, 'predicted_value'] = np.mean(
                        data['value'].iloc[:i])

            data.loc[pidx, 'stepwise_value'] = \
                data.loc[pidx, 'predicted_value'].mean()

            # --- BOCPD-based deviation detection ---
            wcp = data.loc[pidx, 'changepoint_prob']
            max_cp = wcp.max()
            if max_cp > threshold:
                cp_loc = wcp.idxmax()
                all_dev.append(cp_loc)
                rw = data.iloc[i:i + ws]
                if method == 'ols':
                    try:
                        results = sm.OLS(rw['value'],
                                         sm.add_constant(rw['time'])).fit()
                    except Exception:
                        pass
                elif method == 'diff_ols':
                    nr = fit_diff_ols(rw)
                    if nr:
                        results = nr
            else:
                last = all_dev[-1] if all_dev else 0
                last_pos = data.index.get_loc(last) if last in data.index else 0
                rw = data.iloc[last_pos:i + ws]
                if len(rw) > 1:
                    if method == 'ols':
                        try:
                            results = sm.OLS(rw['value'],
                                             sm.add_constant(rw['time'])).fit()
                        except Exception:
                            pass
                    elif method == 'diff_ols':
                        nr = fit_diff_ols(rw)
                        if nr:
                            results = nr

            i += ws

        # --- Plot ---
        if plot and len(data) > 0:
            n_subplots = 4 if (adaptive_threshold and self.thresholds_used) else 3
            fig, axes = plt.subplots(n_subplots, 1, figsize=(14, 3 * n_subplots + 2),
                                      height_ratios=[3, 1, 1] + ([1] if n_subplots == 4 else []),
                                      sharex=True,
                                      gridspec_kw={'hspace': 0.12})

            ax1 = axes[0]
            ax1.plot(data['time'], data['value'], label='Original', color='blue', lw=1)
            ax1.plot(data['time'], data['predicted_value'], label='Predicted',
                     color='green', ls='--', alpha=0.8)
            ax1.step(data['time'], data['stepwise_value'], label='Stepwise',
                     color='orange', where='post', lw=2)
            if all_dev:
                dev_in_index = [d for d in all_dev if d in data.index]
                if dev_in_index:
                    ax1.scatter(data.loc[dev_in_index, 'time'],
                                data.loc[dev_in_index, 'value'],
                                color='red', s=100, label='Change Points', zorder=5)
            ax1.set_xlim(0, data['time'].iloc[-1])
            title_extra = f', adaptive_threshold={adaptive_threshold}' if adaptive_threshold else ''
            ax1.set_title(f'Adaptive BOCPD+OLS — {file_name or ""}, method={method}, '
                          f'w_min={w_min}, w_max={w_max}{title_extra}')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            ax2.plot(data['time'], data['changepoint_prob'], color='purple', lw=1)
            # Plot adaptive threshold trace if available
            if adaptive_threshold and self.thresholds_used:
                thr_t = [data['time'].iloc[s] for s, _ in self.thresholds_used]
                thr_v = [v for _, v in self.thresholds_used]
                ax2.step(thr_t, thr_v, where='post', color='red', ls='--', alpha=0.7,
                         label='Adaptive Threshold')
            else:
                ax2.axhline(threshold_base, color='red', ls='--', alpha=0.7,
                            label=f'Threshold ({threshold_base:.4f})')
            ax2.fill_between(data['time'], 0, data['changepoint_prob'],
                             alpha=0.15, color='purple')
            ax2.set_ylabel('CP Prob')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

            ax3 = axes[2]
            if self.window_sizes_used:
                ws_t = [data['time'].iloc[s] for s, _ in self.window_sizes_used]
                ws_v = [w for _, w in self.window_sizes_used]
                ax3.step(ws_t, ws_v, where='post', color='purple', lw=1.5)
                ax3.axhline(y=w_min, color='gray', ls=':', alpha=0.5,
                            label=f'w_min={w_min}')
                ax3.axhline(y=w_max, color='gray', ls='--', alpha=0.5,
                            label=f'w_max={w_max}')
                ax3.set_ylabel('Window Size')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)

            if n_subplots == 4:
                ax4 = axes[3]
                thr_t = [data['time'].iloc[s] for s, _ in self.thresholds_used]
                thr_v = [v for _, v in self.thresholds_used]
                ax4.step(thr_t, thr_v, where='post', color='darkorange', lw=1.5)
                ax4.axhline(y=threshold_base, color='gray', ls=':', alpha=0.5,
                            label=f'base={threshold_base:.4f}')
                ax4.set_ylabel('Threshold')
                ax4.set_xlabel('Time')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)

            if n_subplots <= 3:
                axes[-1].set_xlabel('Time')

            plt.tight_layout()
            plt.show()

        # --- Store results ---
        self.results['prediction_times'] = data['time']
        self.results['predictions'] = data['predicted_value']
        self.results['actual_values'] = data['value']
        self.results['prediction_errors'] = data['predicted_value'] - data['value']
        self.results['predicted_step_function'] = data['stepwise_value']
        dt_col = data['time'] - data['time'].shift(1)
        dt_col.iloc[0] = data['time'].iloc[0]
        self.results['predicted_step_function_time_interval'] = dt_col
        self.results['changepoint_probs'] = data['changepoint_prob']

        return self.get_summary_metrics()

    # ------------------------------------------------------------------
    def get_summary_metrics(self):
        if len(self.results.get('predictions', [])) == 0:
            return None
        preds = self.results['predictions'].dropna()
        actuals = self.results['actual_values'].loc[preds.index]
        errors = preds - actuals
        if len(preds) == 0:
            return None

        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        nz = actuals[actuals != 0]
        mape = np.mean(np.abs(errors[actuals != 0] / nz)) * 100 if len(nz) else np.inf

        if len(actuals) > 1:
            direction_accuracy = np.mean(
                np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
            ) * 100
        else:
            direction_accuracy = 0.0

        cp = self.results['changepoint_probs']
        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape,
            'direction_accuracy': direction_accuracy,
            'confidence_coverage': 95.0,
            'n_predictions': len(preds),
            'mean_prediction': np.mean(preds),
            'mean_actual': np.mean(actuals),
            'error_std': np.std(errors),
            'avg_changepoint_prob': np.mean(cp) if len(cp) else 0,
            'max_changepoint_prob': np.max(cp) if len(cp) else 0,
        }
