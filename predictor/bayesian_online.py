
# cpd + bayesian online learning - Corrected Incremental Version

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t as t_dist
from .base import RollingPredictor

class BOOLES_cpdPredictor(RollingPredictor):
    def __init__(self, **kwargs):
        self.min_history_length = kwargs['min_history_length']
        self.hazard_lambda = kwargs['hazard_lambda']
        self.mu0 = kwargs['mu0']
        self.kappa0 = kwargs['kappa0']
        self.alpha0 = kwargs['alpha0']
        self.beta0 = kwargs['beta0']
        self.alarm_threshold = kwargs['alarm_threshold']
        self.alarm_min_consecutive = kwargs['alarm_min_consecutive']
        self.results = {
            'predicted_value': [],
            'value': [],
            'prediction_errors': [],
            'stepwise_value': [],
            'predicted_step_function_time_interval': [],
            'changepoint_probs': []
        }
        

        # Online BOCPD state
        self.bocpd_initialized = False
        self.R = None  # Run length probabilities
        self.muT = None  # Mean parameters for each run length
        self.kappaT = None  # Precision parameters
        self.alphaT = None  # Shape parameters  
        self.betaT = None   # Rate parameters
        self.current_time = 0
        self.max_run_length = 1000  # Prevent infinite growth

    def hazard(self, r):
        """Hazard function - probability of changepoint at run length r"""
        # return 1.0 / self.hazard_lambda
        if r == 0:
            return 0.0
        else:
            base_h = 1.0 / self.hazard_lambda
            growth_factor = 1.0 + (r / self.hazard_lambda)
            return min(base_h * growth_factor, 0.5)

    def initialize_bocpd(self):
        """Initialize BOCPD state for online processing"""
        self.R = np.array([1.0])  # Start with probability 1 at run length 0
        self.muT = [self.mu0]
        self.kappaT = [self.kappa0] 
        self.alphaT = [self.alpha0]
        self.betaT = [self.beta0]
        self.current_time = 0
        self.bocpd_initialized = True
    def update_bocpd_online(self, x):
        """
        """
        if not self.bocpd_initialized:
            self.initialize_bocpd()
        
        self.current_time += 1
        
        # Compute the predictive probability for each run length
        current_run_lengths = len(self.R)
        pred_probs = np.zeros(current_run_lengths)
        
        for r in range(current_run_lengths):
            mu = self.muT[r]
            kappa = self.kappaT[r] 
            alpha = self.alphaT[r]
            beta = self.betaT[r]
            
            # Student-t predictive distribution with better numerical stability
            if kappa > 0 and alpha > 0 and beta > 0:
                scale = np.sqrt((beta * (kappa + 1)) / (alpha * kappa))
                df = 2 * alpha
                
                if scale > 1e-10 and df > 0:
                    standardized = (x - mu) / scale
                    standardized = np.clip(standardized, -10, 10)
                    pred_probs[r] = t_dist.pdf(standardized, df) / scale
                else:
                    pred_probs[r] = 1e-10
            else:
                pred_probs[r] = 1e-10
        
        # FIXED: Ensure predictive probabilities are not too small
        pred_probs = np.maximum(pred_probs, 1e-300)
        
        # Calculate hazard probabilities
        run_lengths = np.arange(current_run_lengths)
        hazard_probs = np.array([self.hazard(r) for r in run_lengths])
        
        # Growth probabilities (continue existing run lengths)
        growth_probs = self.R * pred_probs * (1 - hazard_probs)
        
        # Changepoint probability (start new run length)
        cp_prob = np.sum(self.R * pred_probs * hazard_probs)
        
        # Update run length distribution
        new_R = np.zeros(current_run_lengths + 1)
        new_R[0] = cp_prob  # New run length 0 (changepoint)
        new_R[1:] = growth_probs  # Existing run lengths + 1
        
        # FIXED: More robust normalization
        sum_R = np.sum(new_R)
        if sum_R > 1e-300:  # Use a smaller threshold
            new_R = new_R / sum_R
        else:
            # If the sum is too small, reset to initial state
            print(f"Warning: Numerical underflow at time {self.current_time}, sum={sum_R}")
            new_R = np.zeros(current_run_lengths + 1)
            new_R[0] = 1.0  # All probability to changepoint
        
        # FIXED: Prevent probability distribution degradation
        # Add small uniform noise to maintain numerical stability
        epsilon = 1e-10
        new_R = new_R * (1 - epsilon) + epsilon / len(new_R)
        
        # Update sufficient statistics (Bayesian updates)
        new_muT = []
        new_kappaT = []
        new_alphaT = []
        new_betaT = []
        
        # Parameters for new run length (changepoint)
        new_muT.append(self.mu0)
        new_kappaT.append(self.kappa0)
        new_alphaT.append(self.alpha0) 
        new_betaT.append(self.beta0)
        
        # Update parameters for continuing run lengths
        for r in range(current_run_lengths):
            if r < len(self.muT):
                mu = self.muT[r]
                kappa = self.kappaT[r]
                alpha = self.alphaT[r] 
                beta = self.betaT[r]
                
                # Bayesian updates
                kappa_new = kappa + 1
                mu_new = (kappa * mu + x) / kappa_new
                alpha_new = alpha + 0.5
                
                # FIXED: Prevent beta parameter from becoming too large
                beta_increment = (kappa * (x - mu)**2) / (2 * kappa_new)
                beta_new = beta + beta_increment
                
                # FIXED: Limit parameter range to maintain numerical stability
                kappa_new = min(kappa_new, 1e6)
                alpha_new = min(alpha_new, 1e6)
                beta_new = min(beta_new, 1e10)
                
                new_muT.append(mu_new)
                new_kappaT.append(kappa_new)
                new_alphaT.append(alpha_new)
                new_betaT.append(beta_new)
            else:
                new_muT.append(self.mu0)
                new_kappaT.append(self.kappa0)
                new_alphaT.append(self.alpha0)
                new_betaT.append(self.beta0)
        
        # Truncate if necessary
        if len(new_R) > self.max_run_length:
            # Keep the most probable run lengths
            keep_indices = np.argsort(new_R)[-self.max_run_length:]
            new_R = new_R[keep_indices]
            new_R = new_R / np.sum(new_R)  # Re-normalize
            
            # Truncate parameters accordingly
            new_muT = [new_muT[i] if i < len(new_muT) else self.mu0 for i in keep_indices]
            new_kappaT = [new_kappaT[i] if i < len(new_kappaT) else self.kappa0 for i in keep_indices]
            new_alphaT = [new_alphaT[i] if i < len(new_alphaT) else self.alpha0 for i in keep_indices]
            new_betaT = [new_betaT[i] if i < len(new_betaT) else self.beta0 for i in keep_indices]
        
        # Update state
        self.R = new_R
        self.muT = new_muT
        self.kappaT = new_kappaT
        self.alphaT = new_alphaT
        self.betaT = new_betaT
        
        # Return changepoint probability
        return self.R[0]

    def get_current_model_params(self):
        """Get parameters from the most likely run length"""
        if not self.bocpd_initialized or len(self.R) == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0
        
        # Find most probable run length
        most_likely_idx = np.argmax(self.R)
        
        if most_likely_idx < len(self.muT):
            return (self.muT[most_likely_idx], 
                   self.kappaT[most_likely_idx],
                   self.alphaT[most_likely_idx], 
                   self.betaT[most_likely_idx])
        else:
            return self.mu0, self.kappa0, self.alpha0, self.beta0


    def get_current_model_params(self):
        """Get parameters from the most likely run length"""
        if not self.bocpd_initialized or len(self.R) == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0
        
        # Find most probable run length
        most_likely_idx = np.argmax(self.R)
        
        if most_likely_idx < len(self.muT):
            return (self.muT[most_likely_idx], 
                   self.kappaT[most_likely_idx],
                   self.alphaT[most_likely_idx], 
                   self.betaT[most_likely_idx])
        else:
            return self.mu0, self.kappa0, self.alpha0, self.beta0

    def rolling_prediction(self, data, window_size=3, gamma=None, plot=True, method='ols', 
                          file_name=None, cpd_method='bayesian'):
        """
        """
        data = data.copy()
        variability_all = data['value'].std()
        if 'time' not in data.columns:
            data['time'] = np.arange(len(data))
        
        if 'predicted_value' not in data.columns:
            data['predicted_value'] = np.nan
            
        if 'changepoint_prob' not in data.columns:
            data['changepoint_prob'] = np.nan

        all_deviation_points = []
        changepoint_probs = []
        
        if len(data) < 2 * window_size:
            print(f"Warning: Data length ({len(data)}) is less than 2*window_size ({2*window_size})")
            return data, all_deviation_points, np.nan

        detect_window_size = int(window_size * 1)
        predict_window_size = 2 * window_size - detect_window_size
        
        def fit_diff_ols(df_window):
            if len(df_window) < 2:
                return None
            
            delta_t = df_window['time'].diff().mean()
            if pd.isna(delta_t) or delta_t == 0:
                delta_t = 1.0
                
            y_series = df_window['value'].diff() / delta_t
            x_series = df_window['value'].shift(1)
            
            temp_df = pd.DataFrame({'y': y_series, 'x': x_series}).dropna()

            if len(temp_df) < 2:
                return None
                
            Y = temp_df['y']
            X = sm.add_constant(temp_df['x'])
            
            try:
                return sm.OLS(Y, X).fit()
            except:
                return None

        # Initialize model with first window
        initial_window = data.iloc[:window_size]
        if method == 'ols':
            X_init = sm.add_constant(initial_window['time'])
            Y_init = initial_window['value']
            results = sm.OLS(Y_init, X_init).fit()
        elif method == 'diff_ols':
            results = fit_diff_ols(initial_window)
            if results is None:
                print("Could not initialize diff_ols model due to insufficient data.")
                return data, [], np.nan

        # Process each data point online for BOCPD
        print("Running online BOCPD...")
        for idx in range(len(data)):
            value = data.iloc[idx]['value']
            cp_prob = self.update_bocpd_online(value)
            changepoint_probs.append(cp_prob)
            data.loc[data.index[idx], 'changepoint_prob'] = cp_prob
            
            if idx % 50 == 0:  # Progress indicator
                print(f"Processed {idx+1}/{len(data)} points, current CP prob: {cp_prob:.4f}")
        
        print("Running windowed prediction...")
        
        threshold = 0.01
        
        for i in range(2 * window_size, len(data) + 1, window_size):
            predict_start_idx = i - window_size

            predict_end_idx = min(predict_start_idx + predict_window_size, len(data))
            predict_indices = data.index[predict_start_idx:predict_end_idx]

            
            # Make predictions using current model
            if method == 'ols':
                X_predict = sm.add_constant(data.loc[predict_indices, 'time'])
                data.loc[predict_indices, 'predicted_value'] = results.predict(X_predict)
                data.loc[predict_indices, 'predicted_value_ci_lower'], data.loc[predict_indices, 'predicted_value_ci_upper'] = results.get_prediction(X_predict).summary_frame(alpha=0.05)['obs_ci_lower'], results.get_prediction(X_predict).summary_frame(alpha=0.05)['obs_ci_upper']

            elif method == 'diff_ols':
                try:
                    beta_0, beta_1 = results.params
                    a = -beta_1
                    b = -beta_0 / beta_1 if abs(beta_1) > 1e-6 else np.mean(data['value'].iloc[:i-window_size])
                    
                    z0_idx = max(0, predict_start_idx - 1)
                    z0 = data['value'].iloc[z0_idx]
                    t0 = data['time'].iloc[z0_idx]
                    time_diffs = data.loc[predict_indices, 'time'] - t0
                    predicted_values = b + (z0 - b) * np.exp(-a * time_diffs)
                    
                    upper_limit = np.max(data['value'].iloc[:i-window_size])
                    lower_limit = np.min(data['value'].iloc[:i-window_size])
                    predicted_values = np.clip(predicted_values, lower_limit, upper_limit)
                    data.loc[predict_indices, 'predicted_value'] = predicted_values
                except:
                    data.loc[predict_indices, 'predicted_value'] = np.mean(data['value'].iloc[:i-window_size])
                    data.loc[predict_indices, 'predicted_value_ci_lower'], data.loc[predict_indices, 'predicted_value_ci_upper'] = np.mean(data['value'].iloc[:i-window_size]), np.mean(data['value'].iloc[:i-window_size])
            # data.loc[predict_indices, 'stepwise_value'] = data.loc[predict_indices, 'predicted_value'].mean()

            # Check for changepoints in detection window using precomputed probabilities
            detect_indices = data.index[i - window_size : i - window_size + detect_window_size]
            if len(detect_indices) > 0:
                window_cp_probs = data.loc[detect_indices, 'changepoint_prob']
                # print(window_cp_probs)
                max_cp_prob = window_cp_probs.max()
                
                print(f"Step {i}: Max CP probability in window = {max_cp_prob:.4f}")
                
                if max_cp_prob > threshold:
                    cp_location = window_cp_probs.idxmax()
                    all_deviation_points.append(cp_location)
                    print(f"Change point detected at index {cp_location} with probability {max_cp_prob:.4f}")
                    
                    # Model adaptation upon deviation
                    refit_window = data.iloc[i - window_size : i - window_size + detect_window_size]
                    if method == 'ols':
                        X_refit = sm.add_constant(refit_window['time'])
                        Y_refit = refit_window['value']
                        try:
                            results = sm.OLS(Y_refit, X_refit).fit()
                        except:
                            print("OLS refit failed, keeping previous model")
                    elif method == 'diff_ols':
                        new_results = fit_diff_ols(refit_window)
                        if new_results:
                            results = new_results
                else:
                    # Continuous learning (no deviation)
                    last_dev_idx = all_deviation_points[-1] if all_deviation_points else 0
                    refit_window = data.iloc[last_dev_idx:i]
                    if len(refit_window) > 1:
                        if method == 'ols':
                            X_refit = sm.add_constant(refit_window['time'])
                            Y_refit = refit_window['value']
                            try:
                                results = sm.OLS(Y_refit, X_refit).fit()
                            except:
                                print("OLS continuous learning failed, keeping previous model")
                        elif method == 'diff_ols':
                            new_results = fit_diff_ols(refit_window)
                            if new_results:
                                results = new_results
            
        # This means: For the first `window_size` rows in the DataFrame `data`,
        # set 'stepwise_value' and 'predicted_value' columns to the mean of the 'value' column over these same rows.
        data.loc[data.index[:window_size], 'stepwise_value'] = data.loc[data.index[:window_size], 'value'].mean()
        data.loc[data.index[:window_size], 'predicted_value'] = data.loc[data.index[:window_size], 'value'].mean()


        # adaptive stepwise value:
        i = window_size
        
        while i < len(data):
            variability = data['predicted_value'].iloc[i:i+window_size].std()
            adaptive_factor = 1/(1 + variability**2 / variability_all**2)
            end_index = int(min(i + min(1, window_size*adaptive_factor), len(data)))
            data.loc[data.index[i:end_index], 'stepwise_value'] = data.loc[data.index[i:end_index], 'predicted_value'].mean()
            i = end_index
            
            print(f"Step {i}: Stepwise value = {data.loc[data.index[i:end_index], 'stepwise_value'].mean():.4f} | Variability = {variability:.4f} | Adaptive factor = {adaptive_factor:.4f}")
        # Plotting
        if plot and len(data) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Top plot: Time series with predictions and changepoints
            ax1.plot(data['time'], data['value'], label='Original Data', color='blue', linewidth=1)
            ax1.plot(data['time'], data['predicted_value'], label='Predicted Values', 
                    color='green', linestyle='--', alpha=0.8)
            ax1.step(data['time'], data['stepwise_value'], label='Stepwise Function', 
                    color='orange', where='post', linewidth=2)
            ax1.fill_between(data['time'], data['predicted_value_ci_lower'], data['predicted_value_ci_upper'], color='orange', alpha=0.3)

            # Mark deviation points
            if all_deviation_points:
                deviation_times = data.loc[all_deviation_points]['time']
                deviation_values = data.loc[all_deviation_points]['value']
                ax1.scatter(deviation_times, deviation_values, 
                           color='red', s=100, label='Detected Change Points', zorder=5)

            ax1.set_xlim(0, data['time'].iloc[-1])
            ax1.set_title(f'Time Series with Online Change Point Detection\nFile: {file_name}, Method: {method}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Changepoint probabilities
            ax2.plot(data['time'], data['changepoint_prob'], color='purple', linewidth=1)
            ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
            ax2.fill_between(data['time'], 0, data['changepoint_prob'], 
                           where=(data['changepoint_prob'] > threshold), 
                           color='red', alpha=0.3, label='Detected Changes')
            ax2.set_xlim(0, data['time'].iloc[-1])
            ax2.set_title('Changepoint Probabilities Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Changepoint Probability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

        # Store results
        self.results['prediction_times'] = data['time']
        self.results['predictions'] = data['predicted_value']
        self.results['actual_values'] = data['value']
        self.results['prediction_errors'] = data['predicted_value'] - data['value']
        self.results['predicted_step_function'] = data['stepwise_value']
        self.results['predicted_step_function_time_interval'] = data['time'] - data['time'].shift(1)
        self.results['predicted_step_function_time_interval'].iloc[0] = data['time'].iloc[0]
        self.results['changepoint_probs'] = data['changepoint_prob']

        return self.get_summary_metrics()
    
    def get_summary_metrics(self):
        """Calculate prediction performance metrics"""
        if len(self.results['predictions']) == 0:
            return None
            
        predictions = self.results['predictions'].dropna()
        actuals = self.results['actual_values'].iloc[predictions.index]
        errors = predictions - actuals
        
        if len(predictions) == 0:
            return None
        
        # Calculate various metrics
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        
        # Handle division by zero in MAPE calculation
        non_zero_actuals = actuals[actuals != 0]
        non_zero_errors = errors[actuals != 0]
        if len(non_zero_actuals) > 0:
            mape = np.mean(np.abs(non_zero_errors / non_zero_actuals)) * 100
        else:
            mape = np.inf
        
        if len(actuals) > 1:
            actual_diff = np.diff(actuals)
            pred_diff = np.diff(predictions)
            if len(actual_diff) > 0 and len(pred_diff) > 0:
                actual_directions = np.sign(actual_diff)
                pred_directions = np.sign(pred_diff)
                direction_accuracy = np.mean(actual_directions == pred_directions) * 100
            else:
                direction_accuracy = 0
        else:
            direction_accuracy = 0
        
        error_std = np.std(errors)
        confidence_coverage = 95.0  # Placeholder
        
        # Changepoint detection metrics
        cp_probs = self.results['changepoint_probs']
        avg_cp_prob = np.mean(cp_probs) if len(cp_probs) > 0 else 0
        max_cp_prob = np.max(cp_probs) if len(cp_probs) > 0 else 0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'confidence_coverage': confidence_coverage,
            'n_predictions': len(predictions),
            'mean_prediction': np.mean(predictions),
            'mean_actual': np.mean(actuals),
            'error_std': error_std,
            'avg_changepoint_prob': avg_cp_prob,
            'max_changepoint_prob': max_cp_prob,
            'n_changepoints_detected': len([p for p in cp_probs if p > 0.05])
        }

#  

# # cpd + bayesian online learning
# import pandas as pd 
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import rbf_kernel
# import statsmodels.api as sm
# from scipy.stats import t as t_dist  # FIXED: Added missing import with alias to avoid conflicts

# class BOOLES_cpdPredictor():
#     def __init__(self, min_history_length=5, hazard_lambda=100, mu0=0.0, kappa0=1.0, 
#                  alpha0=1.0, beta0=1.0, alarm_threshold=0.7, alarm_min_consecutive=1):
#         self.results = {
#             'predicted_value': [],
#             'value': [],
#             'prediction_errors': [],
#             'stepwise_value': [],
#             'predicted_step_function_time_interval': []
#         }
#         self.hazard_lambda = hazard_lambda
#         self.mu0 = mu0
#         self.kappa0 = kappa0
#         self.alpha0 = alpha0
#         self.beta0 = beta0
#         self.alarm_threshold = alarm_threshold
#         self.alarm_min_consecutive = alarm_min_consecutive

#     def hazard(self, r):
#         """Hazard function - probability of changepoint at run length r"""
#         return 1.0 / self.hazard_lambda

#     def run_bocpd(self, data):
#         """
#         Run Bayesian Online Change Point Detection
#         """
#         T = len(data)
#         R = np.zeros((T+1, T+1))
#         R[0, 0] = 1.0

#         muT, kappaT, alphaT, betaT = [self.mu0], [self.kappa0], [self.alpha0], [self.beta0]
#         cp_probs = []

#         for time_step in range(1, T+1):  # FIXED: Renamed from 't' to avoid conflict with t_dist
#             x = data[time_step-1]
#             pred_probs = np.zeros(time_step)
            
#             # FIXED: Calculate predictive probabilities for each run length
#             for run_len in range(time_step):  # FIXED: Renamed variable to avoid conflict
#                 mu, kappa, alpha, beta = muT[run_len], kappaT[run_len], alphaT[run_len], betaT[run_len]
#                 scale = np.sqrt((beta*(kappa+1))/(alpha*kappa))
#                 df = 2*alpha
#                 pred_probs[run_len] = t_dist.pdf((x-mu)/scale, df) / scale  # FIXED: Use t_dist instead of t

#             run_lengths = np.arange(time_step)
#             growth_probs = R[time_step-1, :time_step] * pred_probs * (1 - self.hazard(run_lengths))
#             cp_prob = np.sum(R[time_step-1, :time_step] * pred_probs * self.hazard(run_lengths))

#             R[time_step, 1:time_step+1] = growth_probs
#             R[time_step, 0] = cp_prob
#             R[time_step, :] /= np.sum(R[time_step, :])

#             cp_probs.append(R[time_step, 0])

#             # Update sufficient statistics
#             new_muT, new_kappaT, new_alphaT, new_betaT = [], [], [], []
#             for run_len in range(time_step):  # FIXED: Consistent variable naming
#                 mu, kappa, alpha, beta = muT[run_len], kappaT[run_len], alphaT[run_len], betaT[run_len]
#                 kappa_new = kappa + 1
#                 mu_new = (kappa*mu + x) / kappa_new
#                 alpha_new = alpha + 0.5
#                 beta_new = beta + (kappa*(x-mu)**2) / (2*kappa_new)
#                 new_muT.append(mu_new)
#                 new_kappaT.append(kappa_new)
#                 new_alphaT.append(alpha_new)
#                 new_betaT.append(beta_new)
                
#             new_muT.insert(0, self.mu0)
#             new_kappaT.insert(0, self.kappa0)
#             new_alphaT.insert(0, self.alpha0)
#             new_betaT.insert(0, self.beta0)

#             muT, kappaT, alphaT, betaT = new_muT, new_kappaT, new_alphaT, new_betaT

#         return np.array(cp_probs)

#     def rolling_prediction(self, data, window_size=3, gamma=None, plot=True, method='ols', 
#                           file_name=None, cpd_method='bayesian'):
#         """
#         Analyze deviations between predicted model values and actual data using Bayesian change point detection.
#         """
#         data = data.copy()

#         if 'time' not in data.columns:
#             data['time'] = np.arange(len(data))
        
#         if 'predicted_value' not in data.columns:
#             data['predicted_value'] = np.nan

#         all_deviation_points = []
        
#         if len(data) < 2 * window_size:
#             print(f"Warning: Data length ({len(data)}) is less than 2*window_size ({2*window_size})")
#             return data, all_deviation_points, np.nan

#         detect_window_size = int(window_size * 1)
#         predict_window_size = 2 * window_size - detect_window_size

#         def fit_diff_ols(df_window):
#             if len(df_window) < 2:
#                 return None
            
#             delta_t = df_window['time'].diff().mean()
#             if pd.isna(delta_t) or delta_t == 0:
#                 delta_t = 1.0
                
#             y_series = df_window['value'].diff() / delta_t
#             x_series = df_window['value'].shift(1)
            
#             temp_df = pd.DataFrame({'y': y_series, 'x': x_series}).dropna()

#             if len(temp_df) < 2:
#                 return None
                
#             Y = temp_df['y']
#             X = sm.add_constant(temp_df['x'])
            
#             try:
#                 return sm.OLS(Y, X).fit()
#             except:
#                 return None

#         # Initial Model Training
#         initial_window = data.iloc[:window_size]
#         if method == 'ols':
#             X_init = sm.add_constant(initial_window['time'])
#             Y_init = initial_window['value']
#             results = sm.OLS(Y_init, X_init).fit()
#         elif method == 'diff_ols':
#             results = fit_diff_ols(initial_window)
#             if results is None:
#                 print("Could not initialize diff_ols model due to insufficient data.")
#                 return data, [], np.nan
            
#         # FIXED: Initialize BOCPD state properly
#         bocpd_history = []
        
#         # Prediction and Detection Loop
#         for i in range(2 * window_size, len(data) + 1, window_size):
#             predict_start_idx = i - window_size
#             predict_end_idx = min(predict_start_idx + predict_window_size, len(data))
#             predict_indices = data.index[predict_start_idx:predict_end_idx]
            
#             # Windowed Prediction
#             if method == 'ols':
#                 X_predict = sm.add_constant(data.loc[predict_indices, 'time'])
#                 data.loc[predict_indices, 'predicted_value'] = results.predict(X_predict)
#             elif method == 'diff_ols':
#                 try:
#                     beta_0, beta_1 = results.params
#                     a = -beta_1
#                     b = -beta_0 / beta_1 if abs(beta_1) > 1e-6 else np.mean(data['value'].iloc[:i-window_size])
                    
#                     z0_idx = max(0, predict_start_idx - 1)
#                     z0 = data['value'].iloc[z0_idx]
#                     t0 = data['time'].iloc[z0_idx]
#                     time_diffs = data.loc[predict_indices, 'time'] - t0
#                     predicted_values = b + (z0 - b) * np.exp(-a * time_diffs)
                    
#                     # Bound predictions
#                     upper_limit = np.max(data['value'].iloc[:i-window_size])
#                     lower_limit = np.min(data['value'].iloc[:i-window_size])
#                     predicted_values = np.clip(predicted_values, lower_limit, upper_limit)
#                     data.loc[predict_indices, 'predicted_value'] = predicted_values
#                 except:
#                     # Fallback to simple mean if diff_ols fails
#                     data.loc[predict_indices, 'predicted_value'] = np.mean(data['value'].iloc[:i-window_size])
                    
#             data.loc[predict_indices, 'stepwise_value'] = data.loc[predict_indices, 'predicted_value'].mean()

#             # FIXED: Distributional Comparison with Bayesian Online change point detection
#             detect_indices = data.index[i - window_size : i - window_size + detect_window_size]
#             if cpd_method == 'bayesian' and len(detect_indices) > 0:
#                 # FIXED: Only run BOCPD on the detection window, not entire dataset
#                 detection_data = data.loc[detect_indices, 'value'].values
#                 try:
#                     cp_probs = self.run_bocpd(detection_data)
#                     # Use the last changepoint probability
#                     prob = cp_probs[-1] if len(cp_probs) > 0 else 0.0
#                     print(f"Step {i}: CP probability = {prob:.4f}")
#                 except Exception as e:
#                     print(f"BOCPD failed: {e}")
#                     prob = 0.0
#             else:
#                 prob = 0.0
                    
#             threshold = 0.05
            
#             # Deviation Detection & Model Adaptation
#             if prob > threshold:
#                 all_deviation_points.append(i - window_size)
#                 print(f"Change point detected at index {i - window_size}")
                
#                 # Model Adaptation upon Deviation
#                 refit_window = data.iloc[i - window_size : i - window_size + detect_window_size]
#                 if method == 'ols':
#                     X_refit = sm.add_constant(refit_window['time'])
#                     Y_refit = refit_window['value']
#                     try:
#                         results = sm.OLS(Y_refit, X_refit).fit()
#                     except:
#                         print("OLS refit failed, keeping previous model")
#                 elif method == 'diff_ols':
#                     new_results = fit_diff_ols(refit_window)
#                     if new_results:
#                         results = new_results
#             else:
#                 # Continuous Learning (No Deviation)
#                 last_dev_idx = all_deviation_points[-1] if all_deviation_points else 0
#                 refit_window = data.iloc[last_dev_idx:i]
#                 if len(refit_window) > 1:  # FIXED: Ensure sufficient data for refitting
#                     if method == 'ols':
#                         X_refit = sm.add_constant(refit_window['time'])
#                         Y_refit = refit_window['value']
#                         try:
#                             results = sm.OLS(Y_refit, X_refit).fit()
#                         except:
#                             print("OLS continuous learning failed, keeping previous model")
#                     elif method == 'diff_ols':
#                         new_results = fit_diff_ols(refit_window)
#                         if new_results:
#                             results = new_results

#         # Initial Data Handling
#         data.loc[data.index[:window_size], 'stepwise_value'] = data.loc[data.index[:window_size], 'value'].mean()
#         data.loc[data.index[:window_size], 'predicted_value'] = data.loc[data.index[:window_size], 'value'].mean()

#         # Plotting
#         if plot and len(data) > 0:
#             plt.figure(figsize=(12, 7))
#             plt.plot(data['time'], data['value'], label='Original Data', color='blue')
#             plt.plot(data['time'], data['predicted_value'], label='Predicted Values', 
#                     color='green', linestyle='--', alpha=0.8)
#             plt.step(data['time'], data['stepwise_value'], label='Stepwise Function', 
#                     color='orange', where='post', linewidth=2)

#             # Mark deviation points
#             if all_deviation_points:
#                 deviation_times = data.iloc[all_deviation_points]['time']
#                 plt.scatter(deviation_times, data.iloc[all_deviation_points]['value'], 
#                            color='red', s=100, label='Change Points', zorder=5)

#             plt.xlim(0, data['time'].iloc[-1])
#             plt.title(f'Time Series with Detected Change Points\nFile: {file_name}, Method: {method}')
#             plt.xlabel('Time')
#             plt.ylabel('Value')
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#             plt.tight_layout()
#             plt.show()

#         self.results['prediction_times'] = data['time']
#         self.results['predictions'] = data['predicted_value']
#         self.results['actual_values'] = data['value']
#         self.results['prediction_errors'] = data['predicted_value'] - data['value']
#         self.results['predicted_step_function'] = data['stepwise_value']
#         self.results['predicted_step_function_time_interval'] = data['time'] - data['time'].shift(1)
#         self.results['predicted_step_function_time_interval'].iloc[0] = data['time'].iloc[0]

#         return self.get_summary_metrics()
    
#     def get_summary_metrics(self):
#         """
#         Calculate prediction performance metrics
#         """
#         if len(self.results['predictions']) == 0:
#             return None
            
#         # FIXED: Handle NaN values properly
#         predictions = self.results['predictions'].dropna()
#         actuals = self.results['actual_values'].iloc[predictions.index]
#         errors = predictions - actuals
        
#         if len(predictions) == 0:
#             return None
        
#         # Calculate various metrics
#         mse = np.mean(errors**2)
#         rmse = np.sqrt(mse)
#         mae = np.mean(np.abs(errors))
        
#         # FIXED: Handle division by zero in MAPE calculation
#         non_zero_actuals = actuals[actuals != 0]
#         non_zero_errors = errors[actuals != 0]
#         if len(non_zero_actuals) > 0:
#             mape = np.mean(np.abs(non_zero_errors / non_zero_actuals)) * 100
#         else:
#             mape = np.inf
        
#         # Direction accuracy
#         if len(actuals) > 1:
#             actual_diff = np.diff(actuals)
#             pred_diff = np.diff(predictions)
#             if len(actual_diff) > 0 and len(pred_diff) > 0:
#                 actual_directions = np.sign(actual_diff)
#                 pred_directions = np.sign(pred_diff)
#                 direction_accuracy = np.mean(actual_directions == pred_directions) * 100
#             else:
#                 direction_accuracy = 0
#         else:
#             direction_accuracy = 0
        
#         # FIXED: Simple confidence intervals
#         error_std = np.std(errors)
#         confidence_coverage = 95.0  # Placeholder since we don't have proper confidence intervals
        
#         return {
#             'mse': mse,
#             'rmse': rmse,
#             'mae': mae,
#             'mape': mape,
#             'direction_accuracy': direction_accuracy,
#             'confidence_coverage': confidence_coverage,
#             'n_predictions': len(predictions),
#             'mean_prediction': np.mean(predictions),
#             'mean_actual': np.mean(actuals),
#             'error_std': error_std
#         }