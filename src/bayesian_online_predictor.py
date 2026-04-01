"""
Bayesian Online Change Point Detection Predictor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
from typing import Dict, List, Optional, Tuple, Union
import warnings


class BayesianOnlinePredictor:
    def __init__(self,
                 hazard_lambda: float = 100.0,
                 mu0: float = 0.0,
                 kappa0: float = 1.0,
                 alpha0: float = 1.0,
                 beta0: float = 1.0,
                 alarm_threshold: float = 0.7,
                 max_run_length: int = 1000):
        self.hazard_lambda = hazard_lambda
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alarm_threshold = alarm_threshold
        self.max_run_length = max_run_length
        self._reset_state()

    def _reset_state(self):
        self.bocpd_initialized = False
        self.R = None
        self.muT = None
        self.kappaT = None
        self.alphaT = None
        self.betaT = None
        self.current_time = 0

    def hazard(self, r: int) -> float:
        if r == 0:
            return 0.0
        base_h = 1.0 / self.hazard_lambda
        return min(base_h * (1.0 + r / self.hazard_lambda), 0.5)

    def _initialize_bocpd(self):
        self.R = np.array([1.0])
        self.muT = [self.mu0]
        self.kappaT = [self.kappa0]
        self.alphaT = [self.alpha0]
        self.betaT = [self.beta0]
        self.current_time = 0
        self.bocpd_initialized = True

    def update(self, x: float) -> float:
        """Update with a new observation; returns change point probability."""
        if not self.bocpd_initialized:
            self._initialize_bocpd()

        self.current_time += 1
        n = len(self.R)

        # Student-t predictive probabilities for each run length
        pred_probs = np.zeros(n)
        for r in range(n):
            mu, kappa, alpha, beta = self.muT[r], self.kappaT[r], self.alphaT[r], self.betaT[r]
            if kappa > 0 and alpha > 0 and beta > 0:
                scale = np.sqrt((beta * (kappa + 1)) / (alpha * kappa))
                df = 2 * alpha
                if scale > 1e-10 and df > 0:
                    pred_probs[r] = t_dist.pdf(np.clip((x - mu) / scale, -10, 10), df) / scale
                else:
                    pred_probs[r] = 1e-10
            else:
                pred_probs[r] = 1e-10
        pred_probs = np.maximum(pred_probs, 1e-300)

        hazard_probs = np.array([self.hazard(r) for r in range(n)])
        growth_probs = self.R * pred_probs * (1 - hazard_probs)
        cp_prob = np.sum(self.R * pred_probs * hazard_probs)

        new_R = np.zeros(n + 1)
        new_R[0] = cp_prob
        new_R[1:] = growth_probs

        total = np.sum(new_R)
        if total > 1e-300:
            new_R /= total
        else:
            warnings.warn(f"Numerical underflow at time {self.current_time}, resetting to change point.")
            new_R = np.zeros(n + 1)
            new_R[0] = 1.0

        epsilon = 1e-10
        new_R = new_R * (1 - epsilon) + epsilon / len(new_R)

        new_muT, new_kappaT, new_alphaT, new_betaT = self._update_sufficient_statistics(x)

        if len(new_R) > self.max_run_length:
            new_R, new_muT, new_kappaT, new_alphaT, new_betaT = self._truncate_state(
                new_R, new_muT, new_kappaT, new_alphaT, new_betaT
            )

        self.R = new_R
        self.muT, self.kappaT, self.alphaT, self.betaT = new_muT, new_kappaT, new_alphaT, new_betaT
        return self.R[0]

    def _update_sufficient_statistics(self, x: float) -> Tuple[List, List, List, List]:
        new_muT = [self.mu0]
        new_kappaT = [self.kappa0]
        new_alphaT = [self.alpha0]
        new_betaT = [self.beta0]

        for r in range(len(self.R)):
            mu, kappa, alpha, beta = self.muT[r], self.kappaT[r], self.alphaT[r], self.betaT[r]
            kappa_new = min(kappa + 1, 1e6)
            mu_new = (kappa * mu + x) / kappa_new
            alpha_new = min(alpha + 0.5, 1e6)
            beta_new = min(beta + (kappa * (x - mu) ** 2) / (2 * kappa_new), 1e10)
            new_muT.append(mu_new)
            new_kappaT.append(kappa_new)
            new_alphaT.append(alpha_new)
            new_betaT.append(beta_new)

        return new_muT, new_kappaT, new_alphaT, new_betaT

    def _truncate_state(self, R, muT, kappaT, alphaT, betaT):
        keep = np.argsort(R)[-self.max_run_length:]
        new_R = R[keep] / R[keep].sum()
        return (new_R,
                [muT[i] for i in keep],
                [kappaT[i] for i in keep],
                [alphaT[i] for i in keep],
                [betaT[i] for i in keep])

    def predict(self) -> Dict[str, float]:
        """Weighted prediction across all run lengths."""
        if not self.bocpd_initialized:
            return {'mean': self.mu0, 'variance': np.inf, 'confidence_interval': (self.mu0, self.mu0)}

        weighted_mean = 0.0
        weighted_var = 0.0
        for r in range(len(self.R)):
            mu, kappa, alpha, beta = self.muT[r], self.kappaT[r], self.alphaT[r], self.betaT[r]
            if kappa > 0 and alpha > 0 and beta > 0:
                weighted_mean += self.R[r] * mu
                weighted_var += self.R[r] * (beta * (kappa + 1)) / (alpha * kappa)

        std_dev = np.sqrt(weighted_var)
        return {
            'mean': weighted_mean,
            'variance': weighted_var,
            'std_dev': std_dev,
            'confidence_interval': (weighted_mean - 1.96 * std_dev, weighted_mean + 1.96 * std_dev)
        }

    def rolling_prediction(self,
                           data: Union[pd.DataFrame, np.ndarray],
                           window_size: int = 3,
                           plot: bool = True,
                           verbose: bool = True) -> Dict:
        """
        Perform rolling prediction on the entire dataset.

        Args:
            data: DataFrame with 'time'/'value' columns, or numpy array
            window_size: Window size for change point detection
            plot: Whether to plot results
            verbose: Print progress

        Returns:
            Dict of performance metrics, with results stored in self.results
        """
        self._reset_state()
        self.results = {
            'predicted_value': [],
            'value': [],
            'prediction_errors': [],
            'stepwise_value': [],
            'predicted_step_function_time_interval': [],
            'changepoint_probs': []
        }

        if isinstance(data, pd.DataFrame):
            values = data['value'].values
            times = data['time'].values if 'time' in data.columns else np.arange(len(data))
        else:
            values = np.asarray(data)
            times = np.arange(len(values))

        if verbose:
            print("Running online BOCPD...")

        cp_probs = []
        for i, x in enumerate(values):
            cp_probs.append(self.update(x))
            if verbose and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(values)} — CP prob: {cp_probs[-1]:.4f}")

        if verbose:
            print("Running windowed prediction...")

        predictions, actuals = [], []
        for i in range(window_size, len(values)):
            pred_value = self.predict()['mean']
            actual_value = values[i]
            predictions.append(pred_value)
            actuals.append(actual_value)
            self.results['stepwise_value'].append(pred_value)
            self.results['predicted_step_function_time_interval'].append(times[i] - times[i - window_size])

            if verbose and (i + 1) % 45 == 0:
                window_max = max(cp_probs[i - window_size:i])
                print(f"  Step {i}: max CP prob in window = {window_max:.4f}")

        self.results['predicted_value'] = predictions
        self.results['value'] = actuals
        self.results['prediction_errors'] = [abs(p - a) for p, a in zip(predictions, actuals)]
        self.results['changepoint_probs'] = cp_probs

        metrics = self._calculate_metrics(predictions, actuals, cp_probs)

        if plot:
            self._plot_results(times, values, predictions, cp_probs)

        return metrics

    def _calculate_metrics(self, predictions, actuals, cp_probs) -> Dict:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        errors = np.abs(predictions - actuals)

        direction_accuracy = 0.0
        if len(predictions) > 1:
            direction_accuracy = np.mean((np.diff(predictions) > 0) == (np.diff(actuals) > 0)) * 100

        return {
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(errors),
            'mape': np.mean(errors / np.abs(actuals)) * 100,
            'direction_accuracy': direction_accuracy,
            'n_predictions': len(predictions),
            'mean_prediction': np.mean(predictions),
            'mean_actual': np.mean(actuals),
            'error_std': np.std(errors),
            'avg_changepoint_prob': np.mean(cp_probs),
            'max_changepoint_prob': np.max(cp_probs),
            'n_changepoints_detected': sum(p > 0.05 for p in cp_probs)
        }

    def _plot_results(self, times, values, predictions, cp_probs):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(times, values, 'b-', label='Actual', alpha=0.7)
        ax1.plot(times[len(times) - len(predictions):], predictions, 'r--', label='Predicted', alpha=0.8)
        ax1.set_title('Bayesian Online Predictor: Predictions vs Actual')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(times, cp_probs, 'g-', label='Change Point Probability', alpha=0.7)
        ax2.axhline(y=self.alarm_threshold, color='r', linestyle='--',
                    label=f'Alarm Threshold ({self.alarm_threshold})')
        ax2.set_title('Change Point Detection Probabilities')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Change Point Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# --- Simple convenience API ---
def simple_bayesian_predict(
    times,
    values,
    hazard_lambda: float = 50.0,
    alarm_threshold: float = 0.1,
    window_size: int = 3,
    plot: bool = True,
    verbose: bool = False,
):
    """
    One-call Bayesian online prediction with sensible defaults.

    Returns:
        dict: { 'predictor', 'summary_metrics', 'detailed_results' }
    """
    times = np.asarray(times)
    values = np.asarray(values)
    mu0 = float(np.mean(values[:min(10, len(values))])) if len(values) else 0.0

    predictor = BayesianOnlinePredictor(
        hazard_lambda=hazard_lambda,
        mu0=mu0,
        alarm_threshold=alarm_threshold,
    )
    df = pd.DataFrame({'time': times, 'value': values})
    metrics = predictor.rolling_prediction(df, window_size=window_size, plot=plot, verbose=verbose)

    return {
        'predictor': predictor,
        'summary_metrics': metrics,
        'detailed_results': predictor.results,
    }
