# utils 
from typing import ValuesView
from predictor import get_predictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from src.pmf import transient_distribution_piecewise
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from pathlib import Path
def rolling_predictor(times, values, method, window_size=3, verbose=True, **kwargs):
    """
    Main function: rolling Kalman filter prediction
    
    For each time point t (starting from the (min_history_length+1)th point), use historical data up to t to predict the value at t.
    
    Args:
        times: Time series
        values: Observation sequence  
        min_history_length: Minimum number of historical data points required for prediction
        verbose: Whether to display detailed information
        
    Returns:
        dict: Contains all prediction results and performance metrics
    """

    if method == 'kalman':
        predictor = get_predictor('rolling_kalman',**kwargs)
        # predictor = get_predictor('rolling_kalman', min_history_length=min_history_length)
        summary_metrics = predictor.rolling_prediction(times, values, verbose=verbose)
        predictor.plot_rolling_predictions(original_times=times, original_values=values)

    elif method == 'cpd':
        predictor = get_predictor('kernel',**kwargs)
        # predictor = get()(min_history_length=min_history_length)
        df = pd.DataFrame({'time': times, 'value': values})
        summary_metrics = predictor.rolling_prediction(df,window_size=window_size, gamma=None, plot=True, method='ols',file_name=None)
    elif method == 'cpd_bayesian':
        predictor = get_predictor('cpd_bayesian',**kwargs)
        # predictor = get_predictor('cpd_bayesian', min_history_length=min_history_length,hazard_lambda=20,mu0=0.0,kappa0=0.01,alpha0=0.01,beta0=0.01,alarm_threshold=0.02,alarm_min_consecutive=1)
        df = pd.DataFrame({'time': times, 'value': values})
        summary_metrics = predictor.rolling_prediction(df,window_size=window_size, gamma=None, plot=True, method='ols',file_name=None)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    if verbose and summary_metrics:
        print("\n" + "="*60)
        print("Prediction performance summary")
        print("="*60)
        print(f"Root mean square error (RMSE): {summary_metrics['rmse']:.4f}")
        print(f"Mean absolute error (MAE): {summary_metrics['mae']:.4f}")
        print(f"Mean absolute percentage error (MAPE): {summary_metrics['mape']:.2f}%")
        print(f"Direction prediction accuracy: {summary_metrics['direction_accuracy']:.1f}%")
        print(f"95% confidence interval coverage: {summary_metrics['confidence_coverage']:.1f}%")
        print(f"Successfully predicted points: {summary_metrics['n_predictions']}")
        print("="*60)
    
    
    
    return {
        'predictor': predictor,
        'summary_metrics': summary_metrics,
        'detailed_results': predictor.results
    }
    
def plot_pmf_overlap(window_sizes, t, file_path, save_path, Z_piece, dt_piece, mu, m,
                    z_initial=80, hist_data_path=None, N=100, show_histogram=True):
    """
    Plot PMFs for different window sizes overlapped with simulation histogram.
    
    Args:
        window_sizes: List of window sizes to compare
        t: Time point for PMF calculation
        file_path: Source file path for reference (not used in current implementation)
        save_path: Directory to save the plot
        Z_piece: Predicted step function values
        dt_piece: Time intervals for step function
        mu: Model parameter mu (service rate)
        m: Model parameter m
        z_initial: Initial queue length (used for histogram file naming)
        hist_data_path: Path to histogram data directory
        N: Maximum number of states for PMF calculation
        show_histogram: Whether to overlay simulation histogram
    """

    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for theory curves
    theory_colors = ['red', 'darkgreen', 'orange', 'purple', 'brown']
    
    # counts = None
    x_hist = None
    kl_data=[]
    if show_histogram and hist_data_path:
        try:
            service_rate = 10 if z_initial == 5 else 100 if z_initial == 80 else mu
            hist_filename = f"for_histogram_CoxM1_Z0{z_initial}_serv{service_rate}_t{int(t)}.pickle"
            hist_path = os.path.join(hist_data_path, hist_filename)
            
            with open(hist_path, 'rb') as f:
                hist_data = pickle.load(f)
            
            counts = hist_data['counts']
            bins = hist_data['bins']
   
           
            if bins[0] != 0:
                bins = np.array(bins, dtype=int)
                interval = max(int(bins[1] - bins[0]), 1)
                bins_before = np.arange(0, bins[0], interval)
                bins = np.concatenate([bins_before, bins])
                counts_before = np.zeros(len(bins_before))
                counts = np.concatenate([counts_before, counts])
            
            
            # Normalize counts to counts
            
            total_count = np.sum(counts)
            if total_count > 0:
                counts = counts / total_count
                x_hist = bins[:-1]  # Use bin starts for x coordinates
                
                # Plot histogram
            ax.bar(bins[:-1], counts,
                    width=bins[1] - bins[0] if len(bins) > 1 else 1,
                    alpha=0.4,
                    color='lightblue',
                    label=f'Simulation t={int(t)}',
                    zorder=1)
            print(f"✓ Loaded histogram: {hist_filename}")
            print(f"  Total count: {total_count}")
            print(f"  Probability sum: {np.sum(counts):.6f}")
            print(f"  Histogram range: [{counts.min():.6f}, {counts.max():.6f}]")
            print(f"  Bins range: [{bins[0]}, {bins[-1]}]")
            print(f'length of bins: {len(bins)}, length of counts: {len(counts)}')
        except FileNotFoundError:
            print(f"Warning: Histogram file not found: {hist_path}")
        except Exception as e:
            print(f"Error loading histogram: {e}")
            import traceback
            traceback.print_exc()
    
    plotted_theory_curves = 0
    
    for idx, ws in enumerate(window_sizes):
        try:
            pt = transient_distribution_piecewise(Z_piece, dt_piece, mu, m, t=t, N=N)
            if pt is None or len(pt) == 0:
                print(f"Warning: Empty PMF for window size {ws}")
                continue
            

            x_vals = np.arange(N)
            y_vals = pt[:N]
            # Determine color
            color = theory_colors[idx] if idx < len(theory_colors) else f'C{idx}'
            
            # Plot PMF curve
            ax.plot(x_vals, y_vals,
                   marker='o',
                   linestyle='-',
                   color=color,
                   label=f'Theory WS={ws}',
                   linewidth=2,
                   markersize=4,
                   zorder=2)
            
            print(f"✓ Plotted PMF for window size {ws}")
            print(f"  PMF sum: {np.sum(pt):.6f}")
            print(f"  Plotted PMF sum: {np.sum(y_vals):.6f}")
            print(f"  PMF range: [{np.min(y_vals):.6f}, {np.max(y_vals):.6f}]")
            print(f"  PMF length: {len(pt)}, Plotted length: {len(y_vals)}")
            
            plotted_theory_curves += 1
            

        
        except Exception as e:
            print(f"Error calculating PMF for window size {ws}: {e}")
            import traceback
            traceback.print_exc()

    # Configure plot appearance
    if plotted_theory_curves == 0:
        print("Warning: No theory curves were plotted")
    
    # Set plot properties
    ax.set_title(f'Queue Length Distribution: Cox/M/1\n'
                f't={int(t)}, Z₀={z_initial}, μ={mu}',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Queue Length (Number of Customers)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"pmf_overlap_t{int(t)}.png"
    full_path = save_dir / filename
   
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()
    bin_centers = (bins[:-1] + bins[1:]) / 2
    new_bin_indices = np.floor(bin_centers + 0.5).astype(int)
    max_new_bin = new_bin_indices.max()
    new_counts = np.zeros(max_new_bin + 1)
    for i, idx in enumerate(new_bin_indices):
        if 0 <= idx <= max_new_bin:
            new_counts[idx] += counts[i]
    new_bins = np.arange(max_new_bin + 2)
    bins = new_bins
    counts = new_counts
        
    if bins[-1] < N:
        bins_after = np.arange(bins[-1] + 1, N + 1)
        counts_after = np.zeros(len(bins_after))
        counts = np.concatenate([counts, counts_after])
        bins = np.concatenate([bins, bins_after])
        
    kl_data.append(compare_pmfs_kl(counts, y_vals, labels=("Simulation", "Prediction"), plot=True))
    
 
    
    print(f"✅ PMF overlap plot saved to {full_path}")
    print("-" * 50)


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate KL divergence KL(P||Q) between two probability distributions.
    
    KL(P||Q) = sum(p_i * log(p_i / q_i))
    
    Args:
        p: True/reference distribution (must sum to 1)
        q: Approximating distribution (must sum to 1)
        epsilon: Small value to avoid log(0) and division by zero
        
    Returns:
        KL divergence value (always >= 0)
    """
    # Ensure inputs are numpy arrays
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Make distributions the same length by padding with zeros
    max_len = max(len(p), len(q))
    if len(p) < max_len:
        p = np.pad(p, (0, max_len - len(p)), mode='constant', constant_values=0)
    if len(q) < max_len:
        q = np.pad(q, (0, max_len - len(q)), mode='constant', constant_values=0)
    
    # Add epsilon to avoid numerical issues
    p = p + epsilon
    q = q + epsilon
    
    # Renormalize after adding epsilon
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate KL divergence
    kl_div = np.sum(p * np.log(p / q))
    
    return kl_div


def compare_pmfs_kl(pmf1: [np.ndarray, list], 
                    pmf2: [np.ndarray, list],
                    labels:[str, str] = ("PMF1", "PMF2"),
                    plot: bool = True):
    """
    Compare two PMFs using KL divergence and optionally visualize them.
    
    Args:
        pmf1: First probability mass function
        pmf2: Second probability mass function
        labels: Labels for the two PMFs
        plot: Whether to plot the PMFs for comparison
        
    Returns:
        Dictionary containing KL divergences and statistics
    """
     
 
    
    # Calculate both directions of KL divergence
    kl_1_to_2 = calculate_kl_divergence(pmf1, pmf2)
    kl_2_to_1 = calculate_kl_divergence(pmf2, pmf1)
    
    # Calculate symmetric KL (average of both directions)
    kl_symmetric = (kl_1_to_2 + kl_2_to_1) / 2
    
    # Calculate Jensen-Shannon divergence (symmetric and bounded)
    m = (pmf1 + pmf2) / 2
    js_divergence = (calculate_kl_divergence(pmf1, m) + calculate_kl_divergence(pmf2, m)) / 2
    
    results = {
        f'KL({labels[0]}||{labels[1]})': kl_1_to_2,
        f'KL({labels[1]}||{labels[0]})': kl_2_to_1,
        'KL_symmetric': kl_symmetric,
        'JS_divergence': js_divergence,
        'pmf1_entropy': -np.sum(pmf1 * np.log(pmf1 + 1e-10)),
        'pmf2_entropy': -np.sum(pmf2 * np.log(pmf2 + 1e-10))
    }
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot PMFs
        ax1 = axes[0]
        x1 = np.arange(len(pmf1))
        x2 = np.arange(len(pmf2))

        ax1.plot(x1, pmf1, 'b-o', label=labels[0], alpha=0.7, markersize=4)
        ax1.plot(x2, pmf2, 'r-s', label=labels[1], alpha=0.7, markersize=4)
        ax1.set_xlabel('State')
        ax1.set_ylabel('Probability')
        ax1.set_title('PMF Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        pmf1_nonzero = pmf1[pmf1 > 0]
        pmf2_nonzero = pmf2[pmf2 > 0]
        x1_nonzero = x1[pmf1 > 0]
        x2_nonzero = x2[pmf2 > 0]
        
        ax2.semilogy(x1_nonzero, pmf1_nonzero, 'b-o', label=labels[0], alpha=0.7, markersize=4)
        ax2.semilogy(x2_nonzero, pmf2_nonzero, 'r-s', label=labels[1], alpha=0.7, markersize=4)
        ax2.set_xlabel('State')
        ax2.set_ylabel('Probability (log scale)')
        ax2.set_title('PMF Comparison (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        info_text = (f'KL({labels[0]}||{labels[1]}) = {kl_1_to_2:.4f}\n'
                    f'KL({labels[1]}||{labels[0]}) = {kl_2_to_1:.4f}\n'
                    f'Symmetric KL = {kl_symmetric:.4f}\n'
                    f'JS Divergence = {js_divergence:.4f}')
        
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    return results
