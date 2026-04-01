"""
Rolling Kalman Filter Predictor

Provides one-step-ahead prediction using a Kalman filter on an
Ornstein–Uhlenbeck (mean-reverting) state-space model.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class RollingPredictor(ABC):
    @abstractmethod
    def rolling_prediction(self, times, values, verbose=True):
        pass

    @abstractmethod
    def get_summary_metrics(self):
        pass

    @abstractmethod
    def plot_rolling_predictions(self, original_times=None, original_values=None,
                                 show_confidence=True, show_errors=True):
        pass


class RollingKalmanPredictor(RollingPredictor):
    """
    Rolling Kalman Predictor.

    For each time point *t*, all historical data up to *t* is used to
    predict the value at *t* via a Kalman filter on an OU model:

        dZ = a (b − Z) dt + σ √Z dW
    """

    def __init__(self, min_history_length=5, kalman_a=0.3, kalman_b=80.0):
        self.min_history_length = min_history_length
        self.default_a = kalman_a
        self.default_b = kalman_b

    # ------------------------------------------------------------------
    # Parameter estimation
    # ------------------------------------------------------------------
    def estimate_parameters_window(self, times, values, a, b):
        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.1

        if len(values) > 3:
            residuals = []
            for i in range(1, len(values)):
                predicted = values[i - 1] + a * (b - values[i - 1]) * dt
                residuals.append(values[i] - predicted)
            residual_var = np.var(residuals) if residuals else 1.0
            sigma = np.sqrt(residual_var / (b * dt)) if b * dt > 0 else 1.0
            sigma = max(0.1, min(3.0, sigma))
        else:
            sigma = 1.0

        return a, b, sigma, dt

    # ------------------------------------------------------------------
    # One-step Kalman prediction
    # ------------------------------------------------------------------
    def kalman_one_step_prediction(self, times, values):
        if len(times) < 2:
            return None

        a, b = self.default_a, self.default_b
        a, b, sigma, dt = self.estimate_parameters_window(times, values, a, b)

        Z_current = values[0]
        P_current = 100.0

        for i in range(1, len(values)):
            Z_pred = Z_current + a * (b - Z_current) * dt
            F = 1 - a * dt
            Q = sigma ** 2 * max(0.1, Z_current) * dt
            P_pred = F * P_current * F + Q

            observation = values[i]
            R = max(1.0, 0.1 * abs(observation))
            S = P_pred + R
            K = P_pred / S

            Z_current = max(0.01, Z_pred + K * (observation - Z_pred))
            P_current = (1 - K) * P_pred

        Z_next = Z_current + a * (b - Z_current) * dt
        F = 1 - a * dt
        Q = sigma ** 2 * max(0.1, Z_current) * dt
        P_next = F * P_current * F + Q

        std_pred = np.sqrt(P_next)
        return {
            'prediction': Z_next,
            'uncertainty': std_pred,
            'confidence_lower': Z_next - 1.96 * std_pred,
            'confidence_upper': Z_next + 1.96 * std_pred,
            'parameters': {'a': a, 'b': b, 'sigma': sigma, 'dt': dt},
        }

    # ------------------------------------------------------------------
    # Rolling prediction
    # ------------------------------------------------------------------
    def rolling_prediction(self, times, values, verbose=True):
        if len(times) != len(values):
            raise ValueError("times and values must have the same length")
        if len(times) < self.min_history_length + 1:
            raise ValueError(f"At least {self.min_history_length + 1} data points required")

        sorted_idx = np.argsort(times)
        times = np.array(times)[sorted_idx]
        values = np.array(values)[sorted_idx]

        self.results = {
            'prediction_times': [],
            'predictions': [],
            'actual_values': [],
            'prediction_errors': [],
            'confidence_lower': [],
            'confidence_upper': [],
            'model_parameters': [],
            'used_history_length': [],
            'predicted_step_function': [],
            'predicted_step_function_time_interval': [],
        }

        if verbose:
            print(f"Rolling prediction: {len(times)} points, "
                  f"min_history={self.min_history_length}")

        self.results['predicted_step_function'] = [values[:self.min_history_length].mean()]
        self.results['predicted_step_function_time_interval'] = [
            times[self.min_history_length - 1] - times[0]
        ]
        last_t = times[self.min_history_length - 1]

        for i in range(self.min_history_length, len(times)):
            pred = self.kalman_one_step_prediction(times[:i], values[:i])
            if pred is None:
                continue

            current_time = times[i]
            actual = values[i]

            self.results['prediction_times'].append(current_time)
            self.results['predicted_step_function'].append(pred['prediction'])
            self.results['predicted_step_function_time_interval'].append(current_time - last_t)
            last_t = current_time

            self.results['predictions'].append(pred['prediction'])
            self.results['actual_values'].append(actual)
            self.results['prediction_errors'].append(actual - pred['prediction'])
            self.results['confidence_lower'].append(pred['confidence_lower'])
            self.results['confidence_upper'].append(pred['confidence_upper'])
            self.results['model_parameters'].append(pred['parameters'])
            self.results['used_history_length'].append(i)

            if verbose and i < self.min_history_length + 5:
                print(f"  t={current_time:.3f}: pred={pred['prediction']:.2f}, "
                      f"actual={actual:.2f}, err={actual - pred['prediction']:.2f}")

        if verbose:
            print(f"  Successful predictions: {len(self.results['predictions'])}")

        return self.get_summary_metrics()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def get_summary_metrics(self):
        if not self.results['predictions']:
            return None

        preds = np.array(self.results['predictions'])
        actuals = np.array(self.results['actual_values'])
        errors = np.array(self.results['prediction_errors'])

        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / actuals)) * 100

        if len(actuals) > 1:
            direction_accuracy = np.mean(
                np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
            ) * 100
        else:
            direction_accuracy = 0.0

        lower = np.array(self.results['confidence_lower'])
        upper = np.array(self.results['confidence_upper'])
        coverage = np.mean((actuals >= lower) & (actuals <= upper)) * 100

        return {
            'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape,
            'direction_accuracy': direction_accuracy,
            'confidence_coverage': coverage,
            'n_predictions': len(preds),
            'mean_prediction': np.mean(preds),
            'mean_actual': np.mean(actuals),
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_rolling_predictions(self, original_times=None, original_values=None,
                                 show_confidence=True, show_errors=True):
        if not self.results['predictions']:
            print("No predictions to plot")
            return

        n_plots = 2 if show_errors else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6 * n_plots))
        if n_plots == 1:
            axes = [axes]

        ax1 = axes[0]
        if original_times is not None and original_values is not None:
            ax1.plot(original_times, original_values, 'lightgray', alpha=0.5,
                     linewidth=1, label='Full series')

        pt = self.results['prediction_times']
        ax1.plot(pt, self.results['actual_values'], 'b-o', lw=2, alpha=0.7,
                 label='Actual', zorder=3)
        ax1.plot(pt, self.results['predictions'], 'r-^', lw=2, alpha=0.8,
                 label='Predicted', zorder=3)

        if show_confidence:
            ax1.fill_between(pt, self.results['confidence_lower'],
                             self.results['confidence_upper'],
                             alpha=0.2, color='red', label='95% CI')

        metrics = self.get_summary_metrics()
        if metrics:
            txt = (f"RMSE: {metrics['rmse']:.3f}\n"
                   f"MAE: {metrics['mae']:.3f}\n"
                   f"Dir acc: {metrics['direction_accuracy']:.1f}%\n"
                   f"CI cov: {metrics['confidence_coverage']:.1f}%")
            ax1.text(0.02, 0.98, txt, transform=ax1.transAxes,
                     va='top', bbox=dict(boxstyle='round', facecolor='wheat'))

        ax1.set_ylabel('Z_t')
        ax1.set_title('Rolling Kalman Prediction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if show_errors:
            ax2 = axes[1]
            errs = self.results['prediction_errors']
            ax2.scatter(pt, errs, color='purple', alpha=0.6, s=20)
            ax2.axhline(0, color='black', ls='--', alpha=0.5)
            mu_e = np.mean(errs)
            sd_e = np.std(errs)
            ax2.axhline(mu_e, color='red', label=f'Mean err: {mu_e:.3f}')
            ax2.axhline(mu_e + 2 * sd_e, color='red', ls=':', label='±2σ')
            ax2.axhline(mu_e - 2 * sd_e, color='red', ls=':')
            ax2.set_ylabel('Error')
            ax2.set_xlabel('Time')
            ax2.set_title('Prediction Error')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_prediction_at_time(self, target_time):
        for i, t in enumerate(self.results['prediction_times']):
            if abs(t - target_time) < 1e-6:
                return {
                    'time': t,
                    'prediction': self.results['predictions'][i],
                    'actual': self.results['actual_values'][i],
                    'error': self.results['prediction_errors'][i],
                    'confidence_lower': self.results['confidence_lower'][i],
                    'confidence_upper': self.results['confidence_upper'][i],
                    'history_length': self.results['used_history_length'][i],
                }
        return None
