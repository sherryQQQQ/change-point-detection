
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
import statsmodels.api as sm
    
def _adaptive_window_size(values_so_far, lambda_=0.94, w_min=3, w_max=15, lookback=20,
                          _running_std_max=None):
    """
    Compute adaptive window size based on rolling std normalized by its running max.

    High volatility (rolling_std near its historical max) -> small window (w_min).
    Low volatility  (rolling_std near zero)                -> large window (w_max).
    Only uses past data (causal, no lookahead).

    Parameters
    ----------
    _running_std_max : float or None
        Pre-computed running maximum of rolling stds.  When supplied by the
        caller (who maintains it incrementally) the O(n) loop is skipped,
        reducing per-call cost from O(n) to O(lookback).
    """
    n = len(values_so_far)
    if n < w_min + 1:
        return w_min

    if _running_std_max is None:
        # O(n) fallback when no cached value is available
        _running_std_max = 0.0
        for j in range(1, n):
            start = max(0, j - lookback)
            s = np.std(values_so_far[start:j + 1])
            if s > _running_std_max:
                _running_std_max = s

    if _running_std_max < 1e-12:
        return w_max

    # Current rolling std (O(lookback))
    start = max(0, n - 1 - lookback)
    rolling_std_now = np.std(values_so_far[start:n])

    v_t = rolling_std_now / _running_std_max  # in [0, 1]
    ws = int(w_min + (w_max - w_min) * (1 - v_t))
    return max(w_min, min(ws, w_max))


def _adaptive_window_size_ewma(values_so_far, lambda_=0.94, w_min=3, w_max=15,
                               span=None):
    """
    Compute adaptive window size using Exponentially Weighted Moving Average
    of absolute returns as a volatility proxy.

    Parameters
    ----------
    values_so_far : array-like
        All observed values up to the current time (causal).
    lambda_ : float
        EWMA decay factor (higher = more memory).
    w_min, w_max : int
        Bounds on window size.
    span : int or None
        If provided, overrides lambda_ via lambda_ = 1 - 2/(span+1).

    Returns
    -------
    int : adaptive window size in [w_min, w_max].
    """
    n = len(values_so_far)
    if n < w_min + 1:
        return w_min

    if span is not None:
        lambda_ = 1.0 - 2.0 / (span + 1)

    # Compute EWMA volatility of absolute differences
    abs_diffs = np.abs(np.diff(values_so_far))
    if len(abs_diffs) == 0:
        return w_max

    ewma_vol = abs_diffs[0]
    ewma_max = ewma_vol
    for d in abs_diffs[1:]:
        ewma_vol = lambda_ * ewma_vol + (1 - lambda_) * d
        if ewma_vol > ewma_max:
            ewma_max = ewma_vol

    if ewma_max < 1e-12:
        return w_max

    v_t = ewma_vol / ewma_max  # in [0, 1]
    ws = int(w_min + (w_max - w_min) * (1 - v_t))
    return max(w_min, min(ws, w_max))


def mmd_statistic(x, y, gamma):
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


def prediction_deviation_analysis(data, window_size=3, gamma=None, plot=True):
    """
    Analyze deviations between predicted model values and actual data using Maximum Mean Discrepancy (MMD).
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing 'value' column with time series data
    window_size : int, optional (default=3)
        Size of sliding window for analysis
    gamma : float, optional (default=None)
        RBF kernel bandwidth parameter. If None, calculated using median heuristic
    plot : bool, optional (default=True)
        Whether to plot the results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with original data, predicted values and deviation points detected
    list
        List of detected significant deviation indices
    """
    # Create a copy to avoid modifying the original data
    data = data.copy()
    
    # Add index column if not exists
    if 'index' not in data.columns:
        data['index'] = range(1, len(data) + 1)
    
    # Initialize predicted values column
    if 'predicted_value' not in data.columns:
        data['predicted_value'] = np.nan
    
    # Store all detected deviation points
    all_deviation_points = []
    
    # Need at least 2*window_size data points to start
    if len(data) < 2 * window_size:
        print(f"Warning: Data length ({len(data)}) is less than 2*window_size ({2*window_size})")
        return data, all_deviation_points
    
    # Initialize the model with first window
    X = data.iloc[:window_size]['index']
    Y = data.iloc[:window_size]['value']
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    
    
    for i in range(2*window_size, len(data), window_size):
        # Use past window to predict current window
        X_predict = sm.add_constant(data.iloc[i-window_size:i]['index'])
        data.loc[data.index[i-window_size:i], 'predicted_value'] = results.predict(X_predict)
        data.loc[data.index[i-window_size:i], 'stepwise_value'] = data.loc[data.index[i-window_size:i], 'predicted_value'].mean()
        # MMD detection - determine if distributions are different
        
        if gamma is None:
            # Use median heuristic for kernel bandwidth
            X_vals = np.vstack([
                data.iloc[i-window_size:i]['value'].values.reshape(-1, 1), 
                data.iloc[i-window_size:i]['predicted_value'].values.reshape(-1, 1)
            ])
            dists = rbf_kernel(X_vals, X_vals)
            gamma_val = 1.0 / np.median(dists[dists > 0])
        else:
            gamma_val = gamma
        
        # Calculate MMD between actual and predicted values
        mmd = mmd_statistic(
            data.iloc[i-window_size:i]['value'], 
            data.iloc[i-window_size:i]['predicted_value'], 
            gamma_val
        )
        
        # Use past data to determine threshold
        threshold = 0.1
        if mmd > threshold:
            # print('test')

            # Record i-window_size as a deviation point
            deviation_point_idx = data.index[i-window_size]
            all_deviation_points.append(deviation_point_idx)
            
            # Refit model with current window for future predictions
            X = data.iloc[i-window_size:i]['index']
            Y = data.iloc[i-window_size:i]['value']
            X = sm.add_constant(X)
            model = sm.OLS(Y, X)
            results = model.fit()
            X_predict = sm.add_constant(data.iloc[i-window_size:i]['index'])
            data.loc[data.index[i-window_size:i], 'predicted_value'] = results.predict(X_predict)
            data.loc[data.index[i-window_size:i], 'stepwise_value'] = data.loc[data.index[i-window_size:i], 'predicted_value'].mean()
        else :
            if all_deviation_points == []:
                index = 0
            else:
                index= all_deviation_points[-1]
            X = data.iloc[index:i]['index']
            Y = data.iloc[index:i]['value']
            X = sm.add_constant(X)
            model = sm.OLS(Y, X)
            results = model.fit()
            
            
    # Plot the results if requested
    if plot and len(data) > 0:
        plt.figure(figsize=(12, 7))
        plt.plot(data['index'], data['value'], label='Original Data', color='blue')
        plt.plot(data['index'], data['predicted_value'], label='Predicted Values', 
                 color='green', linestyle='--', alpha=0.8)
        # Plot stepwise function as a line instead of scatter points for better visualization
        plt.step(data['index'], data['stepwise_value'], label='Stepwise Function', color='orange', where='post', linewidth=2)
        # if all_deviation_points:
        #     deviation_point_indices = [data.index.get_loc(dp) for dp in all_deviation_points 
        #                               if dp in data.index]
        #     plt.scatter(data.iloc[deviation_point_indices]['index'], 
        #                data.iloc[deviation_point_indices]['value'], 
        #                color='red', s=100, marker='o', label='Significant Deviations')
            
        #     for idx in deviation_point_indices:
        #         plt.axvline(x=data.iloc[idx]['index'], color='red', 
        #                    linestyle=':', alpha=0.5)
                
      
        plt.title('Time Series with Detected Prediction Deviations')
        plt.xlabel('Index')
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

