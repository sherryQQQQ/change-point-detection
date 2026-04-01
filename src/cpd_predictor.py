"""
OLS Change Point Detection Predictor

Uses MMD-based distributional comparison to detect regime changes,
then refits OLS or diff-OLS models adaptively.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
import statsmodels.api as sm


def mmd_statistic(x, y, gamma=None):
    """Compute the Maximum Mean Discrepancy between two samples."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)
    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)
    n, m = len(x), len(y)
    mmd = ((np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
           + (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
           - 2 * np.sum(k_xy) / (n * m))
    return mmd


class OLScpdPredictor:
    """
    Change-point detection predictor using OLS / diff-OLS with MMD
    for distributional comparison.
    """

    def __init__(self, min_history_length=5):
        self.min_history_length = min_history_length
        self.results = {
            'predicted_value': [],
            'value': [],
            'prediction_errors': [],
            'stepwise_value': [],
            'predicted_step_function_time_interval': [],
        }

    # ------------------------------------------------------------------
    def rolling_prediction(self, data, window_size=3, gamma=None,
                           plot=True, method='ols', file_name=None,
                           cpd_method='mmd'):
        data = data.copy()
        if 'time' not in data.columns:
            data['time'] = np.arange(len(data))
        if 'predicted_value' not in data.columns:
            data['predicted_value'] = np.nan

        all_deviation_points = []

        if len(data) < 2 * window_size:
            print(f"Warning: data length ({len(data)}) < 2*window_size")
            return data, all_deviation_points, np.nan

        detect_window_size = window_size
        predict_window_size = 2 * window_size - detect_window_size

        # --- helper ---
        def fit_diff_ols(df_window):
            if len(df_window) < 2:
                return None
            delta_t = df_window['time'].diff().mean()
            if pd.isna(delta_t) or delta_t == 0:
                delta_t = 1.0
            y_s = df_window['value'].diff() / delta_t
            x_s = df_window['value'].shift(1)
            tmp = pd.DataFrame({'y': y_s, 'x': x_s}).dropna()
            if len(tmp) < 2:
                return None
            return sm.OLS(tmp['y'], sm.add_constant(tmp['x'])).fit()

        # --- initial fit ---
        init_win = data.iloc[:window_size]
        if method == 'ols':
            results = sm.OLS(init_win['value'],
                             sm.add_constant(init_win['time'])).fit()
        elif method == 'diff_ols':
            results = fit_diff_ols(init_win)
            if results is None:
                print("Could not initialise diff_ols model.")
                return data, [], np.nan

        # --- main loop ---
        for i in range(2 * window_size, len(data) + 1, window_size):
            ps = i - window_size
            pe = ps + predict_window_size
            pidx = data.index[ps:pe]

            if method == 'ols':
                X_p = sm.add_constant(data.loc[pidx, 'time'])
                data.loc[pidx, 'predicted_value'] = results.predict(X_p)
            elif method == 'diff_ols':
                beta_0, beta_1 = results.params
                a = -beta_1
                b = (-beta_0 / beta_1 if abs(beta_1) > 1e-6
                     else np.mean(data['value'].iloc[:i - window_size]))
                z0_idx = max(0, ps - 1)
                z0 = data['value'].iloc[z0_idx]
                t0 = data['time'].iloc[z0_idx]
                td = data.loc[pidx, 'time'] - t0
                pv = b + (z0 - b) * np.exp(-a * td)
                upper = np.max(data['value'].iloc[:i - window_size])
                lower = np.min(data['value'].iloc[:i - window_size])
                pv = np.clip(pv, lower, upper)
                data.loc[pidx, 'predicted_value'] = pv

            data.loc[pidx, 'stepwise_value'] = data.loc[pidx, 'predicted_value'].mean()

            # MMD check
            didx = data.index[i - window_size: i - window_size + detect_window_size]
            if cpd_method == 'mmd':
                prob = mmd_statistic(data.loc[didx, 'value'],
                                     data.loc[didx, 'predicted_value'])
            threshold = 0.03

            if prob > threshold:
                all_deviation_points.append(i - window_size)
                rw = data.iloc[i - window_size: i - window_size + detect_window_size]
                if method == 'ols':
                    results = sm.OLS(rw['value'], sm.add_constant(rw['time'])).fit()
                elif method == 'diff_ols':
                    nr = fit_diff_ols(rw)
                    if nr:
                        results = nr
            else:
                last = all_deviation_points[-1] if all_deviation_points else 0
                rw = data.iloc[last:i]
                if method == 'ols':
                    results = sm.OLS(rw['value'], sm.add_constant(rw['time'])).fit()
                elif method == 'diff_ols':
                    nr = fit_diff_ols(rw)
                    if nr:
                        results = nr

        # fill initial window
        data.loc[data.index[:window_size], 'stepwise_value'] = \
            data.loc[data.index[:window_size], 'value'].mean()
        data.loc[data.index[:window_size], 'predicted_value'] = \
            data.loc[data.index[:window_size], 'value'].mean()

        if plot and len(data) > 0:
            plt.figure(figsize=(12, 7))
            plt.plot(data['time'], data['value'], label='Original', color='blue')
            plt.plot(data['time'], data['predicted_value'], label='Predicted',
                     color='green', ls='--', alpha=0.8)
            plt.step(data['time'], data['stepwise_value'], label='Stepwise',
                     color='orange', where='post', lw=2)
            plt.xlim(0, data['time'].iloc[-1])
            plt.title(f'CPD Prediction — file={file_name}, method={method}')
            plt.xlabel('Time'); plt.ylabel('Value')
            plt.legend(); plt.grid(True, alpha=0.3)
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

        return self.get_summary_metrics()

    # ------------------------------------------------------------------
    def get_summary_metrics(self):
        if len(self.results.get('predictions', [])) == 0:
            return None

        preds_s = self.results['predictions']
        actuals_s = self.results['actual_values']
        valid = preds_s.dropna().index
        preds = np.array(preds_s[valid])
        actuals = np.array(actuals_s[valid])
        errors = preds - actuals

        if len(preds) == 0:
            return None

        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        nz = actuals[actuals != 0]
        mape = np.mean(np.abs((preds[actuals != 0] - nz) / nz)) * 100 if len(nz) else np.inf

        if len(actuals) > 1:
            direction_accuracy = np.mean(
                np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
            ) * 100
        else:
            direction_accuracy = 0.0

        lower = preds - 1.96 * np.abs(errors)
        upper = preds + 1.96 * np.abs(errors)
        coverage = np.mean((actuals >= lower) & (actuals <= upper)) * 100

        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape,
            'direction_accuracy': direction_accuracy,
            'confidence_coverage': coverage,
            'n_predictions': len(preds),
            'mean_prediction': np.mean(preds),
            'mean_actual': np.mean(actuals),
        }
