#kalman filter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
from src.pmf import transient_distribution_piecewise
import os
import pickle
import numpy as np
from pathlib import Path
from .base import RollingPredictor
    
class RollingKalmanPredictor(RollingPredictor):
    def __init__(self, min_history_length=5):
        self.min_history_length = min_history_length
        self.a = a
        self.b = b
 
    def estimate_parameters_window(self, times, values):
        """
        Estimate model parameters from a given time window
        
        Use moment estimation for fast parameter estimation
        """
        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.1
        
        # Long-term mean
        # b = np.mean(values)
        
        # Mean reversion speed (from lag-1 autocorrelation)
        # if len(values) > 2:
        #     values_array = np.array(values)
        #     corr = np.corrcoef(values_array[:-1], values_array[1:])[0, 1]
        #     phi = max(0.05, min(0.95, corr))
        #     a = max(0.01, (1 - phi) / dt)
        # else:
        #     a = 0.3
        
        # Volatility parameter
        if len(values) > 3:
            residuals = []
            for i in range(1, len(values)):
                predicted = values[i-1] + self.a * (self.b - values[i-1]) * dt
                residual = values[i] - predicted
                residuals.append(residual)
            
            if len(residuals) > 0:
                residual_var = np.var(residuals)
                sigma = np.sqrt(residual_var / (b * dt))
                sigma = max(0.1, min(3.0, sigma))
            else:
                sigma = 1.0
        else:
            sigma = 1.0
        
        return a, b, sigma, dt
    
    def kalman_one_step_prediction(self, times, values):
        """
        Use historical data for one-step prediction
        
        Args:
            times: Historical time series
            values: Historical observation sequence
            
        Returns:
            dict: Contains prediction value, uncertainty, etc.
        """
        if len(times) < 2:
            return None
        
        # Estimate model parameters (only using historical data)
        a,b= 0.3,80

        a, b, sigma, dt = self.estimate_parameters_window(times, values,a,b)
        # Initialize Kalman filter
        Z_current = values[0]
        P_current = 100.0
        
        # Update Kalman filter state using historical data
        for i in range(1, len(values)):
            # Prediction step
            Z_pred = Z_current + a * (b - Z_current) * dt
            F = 1 - a * dt
            Q = sigma**2 * max(0.1, Z_current) * dt
            P_pred = F * P_current * F + Q
            
            # Update step
            observation = values[i]
            H = 1.0
            R = max(1.0, 0.1 * abs(observation))
            S = H * P_pred * H + R
            K = P_pred * H / S
            
            innovation = observation - Z_pred
            Z_current = max(0.01, Z_pred + K * innovation)
            P_current = (1 - K * H) * P_pred
        
        # Next step prediction based on current state
        Z_next_pred = Z_current + a * (b - Z_current) * dt
        F = 1 - a * dt
        Q = sigma**2 * max(0.1, Z_current) * dt
        P_next_pred = F * P_current * F + Q
        
        # Calculate confidence interval
        std_pred = np.sqrt(P_next_pred)
        conf_lower = Z_next_pred - 1.96 * std_pred
        conf_upper = Z_next_pred + 1.96 * std_pred
        
        return {
            'prediction': Z_next_pred,
            'uncertainty': std_pred,
            'confidence_lower': conf_lower,
            'confidence_upper': conf_upper,
            'parameters': {'a': a, 'b': b, 'sigma': sigma, 'dt': dt}
        }
    
    def rolling_prediction(self, times, values, verbose=True):
        """
        Execute rolling prediction: for each time point, use previous data to predict the value at that point
        
        Args:
            times: Complete time series
            values: Complete observation sequence
            verbose: Whether to output detailed information
            
        Returns:
            dict: Summary of prediction results
        """
        # Validate input
        if len(times) != len(values):
            raise ValueError("times and values must have the same length")
        
        if len(times) < self.min_history_length + 1:
            raise ValueError(f"At least {self.min_history_length + 1} data points are required")
        
        # Ensure data is sorted by time
        sorted_indices = np.argsort(times)
        times = np.array(times)[sorted_indices]
        values = np.array(values)[sorted_indices]
        
        # Reset results
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
            'predicted_step_function_time_interval': []
        }
        
        if verbose:
            print("="*60)
            print("Rolling prediction")
            print("="*60)
            print(f"Total data points: {len(times)}")
            print(f"Prediction points: {len(times) - self.min_history_length}")
            print(f"Minimum history length: {self.min_history_length}")
        
        # For each time point, predict the value at that point (starting from the (min_history_length+1)th point)
        successful_predictions = 0
        self.results['predicted_step_function'] = [values[:self.min_history_length].mean()]
        self.results['predicted_step_function_time_interval'] = [times[self.min_history_length-1] - times[0]]
        last_current_time = times[self.min_history_length-1]
        for i in range(self.min_history_length, len(times)):
            # Use times[0:i] and values[0:i] to predict values[i]
            history_times = times[:i]
            history_values = values[:i]
            
            # Current time point and actual value to be predicted
            current_time = times[i]
            actual_value = values[i]
            
            # Perform prediction (only using historical data)
            pred_result = self.kalman_one_step_prediction(history_times, history_values)
            
            if pred_result is not None:
                prediction = pred_result['prediction']
                conf_lower = pred_result['confidence_lower']
                conf_upper = pred_result['confidence_upper']
                params = pred_result['parameters']
                
                # Store results
                # if len(self.results['prediction_times']) == 1 or current_time != self.results['prediction_times'][-1]:
                self.results['prediction_times'].append(current_time)
                self.results['predicted_step_function'].append(prediction)
                self.results['predicted_step_function_time_interval'].append(current_time -last_current_time)
                last_current_time = current_time
                
                self.results['predictions'].append(prediction)
                self.results['actual_values'].append(actual_value)
                self.results['prediction_errors'].append(actual_value - prediction)
                self.results['confidence_lower'].append(conf_lower)
                self.results['confidence_upper'].append(conf_upper)
                self.results['model_parameters'].append(params)
                self.results['used_history_length'].append(len(history_times))
                
                successful_predictions += 1
                
                if verbose and i < self.min_history_length + 5:
                    print(f"Time {current_time:.3f}: Predicted={prediction:.2f}, Actual={actual_value:.2f}, "
                          f"Error={actual_value - prediction:.2f}")
        
        if verbose:
            print(f"Successfully predicted: {successful_predictions} points")
        
        return self.get_summary_metrics()
    
    def get_summary_metrics(self):
        """
        Calculate prediction performance metrics
        """
        if len(self.results['predictions']) == 0:
            return None
        
        predictions = np.array(self.results['predictions'])
        actuals = np.array(self.results['actual_values'])
        errors = np.array(self.results['prediction_errors'])
        
        # Calculate various metrics
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / actuals)) * 100
        
        # Direction accuracy (whether the predicted trend is correct)
        if len(actuals) > 1:
            actual_directions = np.sign(np.diff(actuals))
            pred_directions = np.sign(np.diff(predictions))
            direction_accuracy = np.mean(actual_directions == pred_directions) * 100
        else:
            direction_accuracy = 0
        
        # Confidence interval coverage
        lower_bounds = np.array(self.results['confidence_lower'])
        upper_bounds = np.array(self.results['confidence_upper'])
        coverage = np.mean((actuals >= lower_bounds) & (actuals <= upper_bounds)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'confidence_coverage': coverage,
            'n_predictions': len(predictions),
            'mean_prediction': np.mean(predictions),
            'mean_actual': np.mean(actuals)
        }
    
    def plot_rolling_predictions(self, original_times=None, original_values=None, 
                               show_confidence=True, show_errors=True):
        """
        Plot rolling prediction results
        
        Args:
            original_times: Original complete time series (for context)
            original_values: Original complete observation sequence
            show_confidence: Whether to show confidence interval
            show_errors: Whether to show error plot
        """
        if len(self.results['predictions']) == 0:
            print("No prediction results to plot")
            return
        
        n_plots = 2 if show_errors else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 6*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # Main plot: Predicted vs Actual
        ax1 = axes[0]
        
        # If original data is provided, plot the complete observation sequence as background
        if original_times is not None and original_values is not None:
            ax1.plot(original_times, original_values, 'lightgray', alpha=0.5, 
                    linewidth=1, label='Complete observation sequence')
        
        # Prediction and actual points
        pred_times = self.results['prediction_times']
        predictions = self.results['predictions']
        actual_values = self.results['actual_values']
        # Use line plots instead of scatter plots for actual and predicted values
        ax1.plot(pred_times, actual_values, color='blue', alpha=0.7, linewidth=2, 
                 label='Actual values', zorder=3, marker='o')
        ax1.plot(pred_times, predictions, color='red', alpha=0.8, linewidth=2, 
                 label='Predicted values', zorder=3, marker='^')
        
        # Connecting lines to show the relationship between predicted and actual values
        for i in range(len(pred_times)):
            ax1.plot([pred_times[i], pred_times[i]], 
                    [actual_values[i], predictions[i]], 
                    'gray', alpha=0.3, linewidth=1, zorder=1)
        
        # Confidence interval
        if show_confidence:
            conf_lower = self.results['confidence_lower']
            conf_upper = self.results['confidence_upper']
            ax1.fill_between(pred_times, conf_lower, conf_upper, 
                           alpha=0.2, color='red', label='95% Confidence interval')
        
        ax1.set_ylabel('Intensity value Z_t')
        ax1.set_title('Rolling prediction result: predict each time point using historical data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add performance metrics text
        metrics = self.get_summary_metrics()
        if metrics:
            metrics_text = (f'RMSE: {metrics["rmse"]:.3f}\n'
                           f'MAE: {metrics["mae"]:.3f}\n'
                           f'Direction accuracy: {metrics["direction_accuracy"]:.1f}%\n'
                           f'Confidence coverage: {metrics["confidence_coverage"]:.1f}%')
            ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Error plot
        if show_errors:
            ax2 = axes[1]
            errors = self.results['prediction_errors']
            
            ax2.scatter(pred_times, errors, color='purple', alpha=0.6, s=20)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Error statistics line
            error_mean = np.mean(errors)
            error_std = np.std(errors)
            ax2.axhline(y=error_mean, color='red', linestyle='-', alpha=0.7, 
                       label=f'Average error: {error_mean:.3f}')
            ax2.axhline(y=error_mean + 2*error_std, color='red', linestyle=':', 
                       alpha=0.7, label='±2 Std')
            ax2.axhline(y=error_mean - 2*error_std, color='red', linestyle=':', alpha=0.7)
            
            ax2.set_ylabel('Prediction error (Actual - Predicted)')
            ax2.set_xlabel('Time')
            ax2.set_title('Prediction error analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_prediction_at_time(self, target_time):
        """
        Get prediction results at a specific time point
        """
        for i, time in enumerate(self.results['prediction_times']):
            if abs(time - target_time) < 1e-6:
                return {
                    'time': time,
                    'prediction': self.results['predictions'][i],
                    'actual': self.results['actual_values'][i],
                    'error': self.results['prediction_errors'][i],
                    'confidence_lower': self.results['confidence_lower'][i],
                    'confidence_upper': self.results['confidence_upper'][i],
                    'history_length': self.results['used_history_length'][i]
                }
        return None


    