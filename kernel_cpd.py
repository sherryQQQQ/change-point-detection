import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

def compute_mmd_score(X: np.ndarray, 
                     w: int = 20, 
                     gamma: float = 1.0) -> np.ndarray:
    """
    Compute MMD-based change point detection scores
    
    Parameters:
        X: Time series data
        w: Window size
        gamma: Parameter for RBF kernel
    
    Returns:
        scores: MMD scores for each time point
    """
    T = len(X)
    scores = np.zeros(T)
    
    for t in range(w, T - w):
        L = X[t - w:t]
        R = X[t:t + w]
        
        # Compute kernel matrices
        K_LL = rbf_kernel(L.reshape(-1, 1), L.reshape(-1, 1), gamma=gamma)
        K_RR = rbf_kernel(R.reshape(-1, 1), R.reshape(-1, 1), gamma=gamma)
        K_LR = rbf_kernel(L.reshape(-1, 1), R.reshape(-1, 1), gamma=gamma)
        
        # Compute unbiased MMD²
        mmd = (np.sum(K_LL) - np.trace(K_LL)) / (w * (w - 1))
        mmd += (np.sum(K_RR) - np.trace(K_RR)) / (w * (w - 1))
        mmd -= 2 * np.mean(K_LR)
        
        scores[t] = mmd
    
    return scores

def detect_change_points(scores: np.ndarray, 
                        threshold: Optional[float] = None,
                        min_distance: int = 50) -> List[int]:
    """
    Detect change point locations
    
    Parameters:
        scores: MMD score sequence
        threshold: Detection threshold, if None use local peaks
        min_distance: Minimum distance between change points
    
    Returns:
        change_points: List of detected change point locations
    """
    if threshold is not None:
        # Use threshold-based detection
        change_points = np.where(scores > threshold)[0]
    else:
        # Use local peak detection
        change_points = []
        for i in range(1, len(scores) - 1):
            if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                change_points.append(i)
        change_points = np.array(change_points)
    
    # Ensure minimum distance between change points
    if len(change_points) > 1:
        filtered_points = [change_points[0]]
        for point in change_points[1:]:
            if point - filtered_points[-1] >= min_distance:
                filtered_points.append(point)
        change_points = filtered_points
    
    return change_points

def plot_change_points(X: np.ndarray, 
                      scores: np.ndarray,
                      change_points: List[int],
                      title: str = "Kernel Change Point Detection") -> None:
    """
    Visualize change point detection results
    
    Parameters:
        X: Original time series
        scores: MMD score sequence
        change_points: Detected change point locations
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Plot original data
    ax1.plot(X, label='Original Data')
    for cp in change_points:
        ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
    ax1.set_title(title)
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MMD scores
    ax2.plot(scores, label='MMD Score')
    ax2.set_ylabel('MMD Score')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def kernel_cpd(X: np.ndarray,
              w: int = 50,
              gamma: float = 1.0,
              threshold: Optional[float] = None,
              min_distance: int = 50,
              plot: bool = True) -> Tuple[np.ndarray, List[int]]:
    """
    Complete kernel-based change point detection pipeline
    
    Parameters:
        X: Time series data
        w: Window size
        gamma: Parameter for RBF kernel
        threshold: Detection threshold
        min_distance: Minimum distance between change points
        plot: Whether to plot results
    
    Returns:
        scores: MMD score sequence
        change_points: Detected change point locations
    """
    # Compute MMD scores
    scores = compute_mmd_score(X, w, gamma)
    
    # Detect change points
    change_points = detect_change_points(scores, threshold, min_distance)
    
    # Visualize results
    if plot:
        plot_change_points(X, scores, change_points)
    
    return scores, change_points

if __name__ == "__main__":
    import pandas as pd
    
    # Read data
    data5 = pd.read_csv('/Users/qianxinhui/Desktop/NU-Research/change-point-detection/data/data5.csv').values.flatten()
    data80 = pd.read_csv('/Users/qianxinhui/Desktop/NU-Research/change-point-detection/data/data80.csv').values.flatten()
    
    # Perform change point detection on both datasets
    print("Detecting change points for Z(0)=5:")
    scores5, cps5 = kernel_cpd(data5, w=50, gamma=0.1)
    
    print("\nDetecting change points for Z(0)=80:")
    scores80, cps80 = kernel_cpd(data80, w=50, gamma=0.1)