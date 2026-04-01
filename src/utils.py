"""
Utility Functions for Prediction and Analysis

This module contains utility functions for metrics calculation, PMF plotting,
and other analysis tasks.

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings



def calculate_prediction_metrics(predictions: Union[np.ndarray, List],
                                actuals: Union[np.ndarray, List],
                                changepoint_probs: Optional[Union[np.ndarray, List]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive prediction performance metrics
    
    Args:
        predictions: Predicted values
        actuals: Actual observed values
        changepoint_probs: Optional change point probabilities
        
    Returns:
        Dictionary containing various performance metrics
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)
    
    if len(predictions) != len(actuals):
        raise ValueError("Predictions and actuals must have the same length")
    
    # Basic error metrics
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(abs_errors)
    mape = np.mean(abs_errors / np.abs(actuals)) * 100
    
    # Direction accuracy
    if len(predictions) > 1:
        pred_directions = np.diff(predictions) > 0
        actual_directions = np.diff(actuals) > 0
        direction_accuracy = np.mean(pred_directions == actual_directions) * 100
    else:
        direction_accuracy = 0.0
    
    # Confidence interval coverage (simplified calculation)
    error_std = np.std(errors)
    confidence_coverage = 95.0  # Placeholder - would need actual confidence intervals
    
    # Change point statistics
    avg_changepoint_prob = 0.0
    max_changepoint_prob = 0.0
    n_changepoints_detected = 0
    
    if changepoint_probs is not None:
        cp_probs = np.asarray(changepoint_probs)
        avg_changepoint_prob = np.mean(cp_probs)
        max_changepoint_prob = np.max(cp_probs)
        n_changepoints_detected = len(cp_probs[cp_probs > 0.05])
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'direction_accuracy': direction_accuracy,
        'confidence_coverage': confidence_coverage,
        'n_predictions': len(predictions),
        'mean_prediction': np.mean(predictions),
        'mean_actual': np.mean(actuals),
        'error_std': error_std,
        'avg_changepoint_prob': avg_changepoint_prob,
        'max_changepoint_prob': max_changepoint_prob,
        'n_changepoints_detected': n_changepoints_detected
    }


def calculate_kl_divergence(p: Union[np.ndarray, List], 
                           q: Union[np.ndarray, List], 
                           epsilon: float = 1e-10) -> float:
    """
    Calculate KL divergence KL(P||Q) between two probability distributions
    
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


def compare_pmfs_kl(pmf1: Union[np.ndarray, List], 
                    pmf2: Union[np.ndarray, List],
                    labels: Tuple[str, str] = ("PMF1", "PMF2"),
                    plot: bool = True) -> Dict[str, float]:
    """
    Compare two PMFs using KL divergence and optionally visualize them
    
    Args:
        pmf1: First probability mass function
        pmf2: Second probability mass function
        labels: Labels for the two PMFs
        plot: Whether to plot the PMFs for comparison
        
    Returns:
        Dictionary containing KL divergences and statistics
    """
    pmf1 = np.asarray(pmf1)
    pmf2 = np.asarray(pmf2)
    
    # Calculate KL divergences in both directions
    kl_1_to_2 = calculate_kl_divergence(pmf1, pmf2)
    kl_2_to_1 = calculate_kl_divergence(pmf2, pmf1)
    
    # Calculate symmetric KL divergence (Jensen-Shannon distance)
    js_distance = 0.5 * (kl_1_to_2 + kl_2_to_1)
    
    # Calculate statistics
    stats = {
        'kl_1_to_2': kl_1_to_2,
        'kl_2_to_1': kl_2_to_1,
        'js_distance': js_distance,
        'pmf1_sum': np.sum(pmf1),
        'pmf2_sum': np.sum(pmf2),
        'pmf1_max': np.max(pmf1),
        'pmf2_max': np.max(pmf2)
    }
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x1 = np.arange(len(pmf1))
        x2 = np.arange(len(pmf2))
        
        ax.plot(x1, pmf1, 'b-', label=labels[0], linewidth=2, marker='o', markersize=4)
        ax.plot(x2, pmf2, 'r--', label=labels[1], linewidth=2, marker='s', markersize=4)
        
        ax.set_title(f'PMF Comparison: {labels[0]} vs {labels[1]}')
        ax.set_xlabel('State')
        ax.set_ylabel('Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add KL divergence info to plot
        ax.text(0.02, 0.98, f'KL({labels[0]}||{labels[1]}): {kl_1_to_2:.4f}\n'
                            f'KL({labels[1]}||{labels[0]}): {kl_2_to_1:.4f}\n'
                            f'JS Distance: {js_distance:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    return stats


def plot_pmf_overlap(window_sizes: List[int],
                    t: float,
                    file_path: str,
                    save_path: str,
                    Z_piece: np.ndarray,
                    dt_piece: np.ndarray,
                    mu: float,
                    m: int,
                    z_initial: int = 80,
                    hist_data_path: Optional[str] = None,
                    N: int = 100,
                    show_histogram: bool = True) -> List[Dict[str, float]]:
    """
    Plot PMFs for different window sizes overlapped with simulation histogram
    
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
        
    Returns:
        List of KL divergence statistics for each window size
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for theory curves
    theory_colors = ['red', 'darkgreen', 'orange', 'purple', 'brown']
    
    counts = None
    x_hist = None
    kl_data = []
    
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
            
            # Normalize counts to probabilities
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
                print(f'  Length of bins: {len(bins)}, length of counts: {len(counts)}')
                
        except FileNotFoundError:
            print(f"Warning: Histogram file not found: {hist_path}")
        except Exception as e:
            print(f"Error loading histogram: {e}")
            import traceback
            traceback.print_exc()
    
    plotted_theory_curves = 0

    from .pmf import transient_distribution_piecewise

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
            
            # Calculate KL divergence if histogram is available
            if counts is not None:
                # Align histogram and PMF
                bin_centers = (bins[:-1] + bins[1:]) / 2
                new_bin_indices = np.floor(bin_centers + 0.5).astype(int)
                max_new_bin = new_bin_indices.max()
                new_counts = np.zeros(max_new_bin + 1)
                
                for i, bin_idx in enumerate(new_bin_indices):
                    if 0 <= bin_idx <= max_new_bin:
                        new_counts[bin_idx] += counts[i]
                
                new_bins = np.arange(max_new_bin + 2)
                bins_aligned = new_bins
                counts_aligned = new_counts
                
                # Extend to match PMF length
                if bins_aligned[-1] < N:
                    bins_after = np.arange(bins_aligned[-1] + 1, N + 1)
                    counts_after = np.zeros(len(bins_after))
                    counts_aligned = np.concatenate([counts_aligned, counts_after])
                    bins_aligned = np.concatenate([bins_aligned, bins_after])
                
                # Calculate KL divergence
                kl_stats = compare_pmfs_kl(counts_aligned, y_vals, 
                                         labels=("Simulation", f"Prediction WS={ws}"), 
                                         plot=False)
                kl_data.append(kl_stats)
            
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
    
    # Save plot
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"pmf_overlap_t{int(t)}.png"
    full_path = save_dir / filename
    
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ PMF overlap plot saved to {full_path}")
    print("-" * 50)
    
    return kl_data


def create_prediction_report(results: Dict[str, Any],
                           method: str,
                           save_path: Optional[str] = None) -> str:
    """
    Create a comprehensive prediction report
    
    Args:
        results: Prediction results dictionary
        method: Prediction method used
        save_path: Optional path to save report
        
    Returns:
        Report text
    """
    metrics = results.get('summary_metrics', {})
    detailed_results = results.get('detailed_results', {})
    
    report = f"""
PREDICTION REPORT
=================
Method: {method.upper()}
Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PERFORMANCE METRICS
-------------------
Root Mean Square Error (RMSE): {metrics.get('rmse', 0):.4f}
Mean Absolute Error (MAE): {metrics.get('mae', 0):.4f}
Mean Absolute Percentage Error (MAPE): {metrics.get('mape', 0):.2f}%
Direction Prediction Accuracy: {metrics.get('direction_accuracy', 0):.1f}%
95% Confidence Interval Coverage: {metrics.get('confidence_coverage', 0):.1f}%

PREDICTION STATISTICS
--------------------
Number of Predictions: {metrics.get('n_predictions', 0)}
Mean Predicted Value: {metrics.get('mean_prediction', 0):.4f}
Mean Actual Value: {metrics.get('mean_actual', 0):.4f}
Error Standard Deviation: {metrics.get('error_std', 0):.4f}
"""
    
    if method == 'bayesian':
        report += f"""
CHANGE POINT DETECTION
----------------------
Average Change Point Probability: {metrics.get('avg_changepoint_prob', 0):.4f}
Maximum Change Point Probability: {metrics.get('max_changepoint_prob', 0):.4f}
Number of Change Points Detected: {metrics.get('n_changepoints_detected', 0)}
"""
    
    if detailed_results:
        report += f"""
DETAILED RESULTS SUMMARY
------------------------
Predicted Values Range: [{np.min(detailed_results.get('predicted_value', [0])):.4f}, {np.max(detailed_results.get('predicted_value', [0])):.4f}]
Actual Values Range: [{np.min(detailed_results.get('value', [0])):.4f}, {np.max(detailed_results.get('value', [0])):.4f}]
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {save_path}")
    
    return report


def load_prediction_data(filepath: str) -> pd.DataFrame:
    """
    Load prediction data from CSV file
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with prediction data
    """
    try:
        data = pd.read_csv(filepath)
        print(f"✓ Loaded prediction data from {filepath}")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        raise


def save_prediction_data(data: Union[pd.DataFrame, Dict[str, Any]], 
                        filepath: str):
    """
    Save prediction data to CSV file
    
    Args:
        data: Data to save (DataFrame or dictionary)
        filepath: Path to save file
    """
    try:
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame(data)
        else:
            df = data
        
        df.to_csv(filepath, index=False)
        print(f"✓ Saved prediction data to {filepath}")
        print(f"  Shape: {df.shape}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        raise
