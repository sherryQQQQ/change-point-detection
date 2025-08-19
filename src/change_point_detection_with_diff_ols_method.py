
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
import statsmodels.api as sm
    
def mmd_statistic(x, y, gamma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two samples.
    
    Parameters:
    -----------
    x : array-like
        First sample
    y : array-like
        Second sample
    gamma : float
        RBF kernel bandwidth parameter
        
    Returns:
    --------
    float
        MMD statistic value
    """
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Compute kernel matrices
    k_xx = rbf_kernel(x, x, gamma)
    k_yy = rbf_kernel(y, y, gamma)
    k_xy = rbf_kernel(x, y, gamma)
    
    # Compute MMD
    n = len(x)
    m = len(y)
    
    mmd = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
    mmd += (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
    mmd -= 2 * np.sum(k_xy) / (n * m)
    
    return mmd

def prediction_deviation_analysis(data, window_size=3, gamma=None, plot=True, method='ols',file_name=None):
    """
    Analyze deviations between predicted model values and actual data using Maximum Mean Discrepancy (MMD).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing 'value' and 'time' columns.
    window_size : int, optional (default=3)
        Size of sliding window for analysis.
    gamma : float, optional (default=None)
        RBF kernel bandwidth parameter. If None, calculated using median heuristic.
    plot : bool, optional (default=True)
        Whether to plot the results.
    method : str, optional (default='ols')
        'ols' for standard linear regression on time, 'diff_ols' for the mean-reverting model.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data, predicted values, and deviation points detected.
    list
        List of detected significant deviation indices.
    """
    data = data.copy()

    if 'time' not in data.columns:
        data['time'] = np.arange(len(data))
    
    if 'predicted_value' not in data.columns:
        data['predicted_value'] = np.nan

    all_deviation_points = []
    
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
        
        return sm.OLS(Y, X).fit()

    # --- Initial Model Training ---
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
    
    # --- Prediction and Detection Loop ---
    for i in range(2 * window_size, len(data) + 1, window_size):
        predict_start_idx = i - window_size
        predict_end_idx = predict_start_idx + predict_window_size
        predict_indices = data.index[predict_start_idx:predict_end_idx]
        
        # --- Windowed Prediction ---
        if method == 'ols':
            X_predict = sm.add_constant(data.loc[predict_indices, 'time'])
            data.loc[predict_indices, 'predicted_value'] = results.predict(X_predict)
        elif method == 'diff_ols':
            beta_0, beta_1 = results.params
            a = -beta_1
            b = -beta_0 / beta_1 if abs(beta_1) > 1e-6 else np.mean(data['value'].iloc[:i-window_size])
            
            z0_idx = predict_start_idx - 1 if predict_start_idx > 0 else 0
            z0 = data['value'].iloc[z0_idx]
            t0 = data['time'].iloc[z0_idx]
            time_diffs = data.loc[predict_indices, 'time'] - t0
            predicted_values = b + (z0 - b) * np.exp(- a * time_diffs)
            
            # Set an upper limit for predicted values
            upper_limit = np.max(data['value'].iloc[:i-window_size])  # You can adjust this logic as needed
            predicted_values = np.minimum(predicted_values, upper_limit)
            lower_limit = np.min(data['value'].iloc[:i-window_size])
            predicted_values = np.maximum(predicted_values, lower_limit)
            data.loc[predict_indices, 'predicted_value'] = predicted_values
        data.loc[predict_indices, 'stepwise_value'] = data.loc[predict_indices, 'predicted_value'].mean()
        
        # --- Distributional Comparison with MMD ---
        detect_indices = data.index[i - window_size : i - window_size + detect_window_size]
        # if method == 'ols':
        #     if gamma is None:
        #         # Use median heuristic for kernel bandwidth
        #         X_vals = np.vstack([
        #             data.iloc[i-window_size:i-window_size+detect_window_size]['value'].values.reshape(-1, 1), 
        #             data.iloc[i-window_size:i-window_size+detect_window_size]['predicted_value'].values.reshape(-1, 1)
        #         ])
        #         dists = rbf_kernel(X_vals, X_vals)
        #         gamma_val = 1.0 / np.median(dists[dists > 0])
        #     else:
        #         gamma_val = gamma
        # --- MMD Calculation ---
        mmd = mmd_statistic(data.loc[detect_indices, 'value'], data.loc[detect_indices, 'predicted_value'])
        threshold = 0.01
        # --- Deviation Detection & Model Adaptation ---
        if mmd > threshold:
            all_deviation_points.append(i - window_size)
            # --- Model Adaptation upon Deviation ---
            refit_window = data.iloc[i - window_size : i - window_size + detect_window_size]
            if method == 'ols':
                X_refit = sm.add_constant(refit_window['time'])
                Y_refit = refit_window['value']
                results = sm.OLS(Y_refit, X_refit).fit()
            elif method == 'diff_ols':
                new_results = fit_diff_ols(refit_window)
                if new_results:
                    results = new_results
        else:
            # --- Continuous Learning (No Deviation) ---
            last_dev_idx = all_deviation_points[-1] if all_deviation_points else 0
            refit_window = data.iloc[last_dev_idx:i]
            if method == 'ols':
                X_refit = sm.add_constant(refit_window['time'])
                Y_refit = refit_window['value']
                results = sm.OLS(Y_refit, X_refit).fit()
            elif method == 'diff_ols':
                new_results = fit_diff_ols(refit_window)
                if new_results:
                    results = new_results

    # --- Initial Data Handling ---
    data.loc[data.index[:window_size], 'stepwise_value'] = data.loc[data.index[:window_size], 'value'].mean()
    data.loc[data.index[:window_size], 'predicted_value'] = data.loc[data.index[:window_size], 'value'].mean()

    # --- Plotting and Final Metrics ---

    # Plot the results if requested
    if plot and len(data) > 0:
        plt.figure(figsize=(12, 7))
        plt.plot(data['time'], data['value'], label='Original Data', color='blue')
        plt.plot(data['time'], data['predicted_value'], label='Predicted Values', 
                 color='green', linestyle='--', alpha=0.8)
        # Plot stepwise function as a line instead of scatter points for better visualization
        plt.step(data['time'], data['stepwise_value'], label='Stepwise Function', color='orange', where='post', linewidth=2)
        # if all_deviation_points:
        #     deviation_point_indices = [data.index.get_loc(dp) for dp in all_deviation_points 
        #                               if dp in data.index]
        #     plt.scatter(data.iloc[deviation_point_indices]['index'], 
        #                data.iloc[deviation_point_indices]['value'], 
        #                color='red', s=100, marker='o', label='Significant Deviations')
            
        #     for idx in deviation_point_indices:
        #         plt.axvline(x=data.iloc[idx]['index'], color='red', 
        #                    linestyle=':', alpha=0.5)
                
        plt.xlim(0, data['time'].iloc[-1])
        plt.title(f'Time Series with Detected Prediction Deviations, file_name={file_name},method={method}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    #calculate mse loss
    MSE = np.mean((data['value'] - data['predicted_value'])**2)
    print('MSE for predicted value', MSE)
    MSE_stepwise = np.mean((data['value'] - data['stepwise_value'])**2)
    print('MSE for stepwise value', MSE_stepwise)
    
    # calculate the kl divergence between the predicted value and the stepwise value
    kl_divergence = np.sum(data['predicted_value'] * np.log(data['predicted_value'] / data['stepwise_value']))
    print('KL divergence', kl_divergence)

    
    return data, all_deviation_points, MSE

