import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats
from scipy.stats import poisson
# w: window size of each interval
 
class KernelCPD:
    def __init__(self, w: int = 20, gamma: float = 1.0, process_type: str = 'normal'):
        """
        Initialize the Kernel Change Point Detection class
        
        Parameters:
            w: Window size for MMD computation, I set it to 20
            gamma: Parameter for RBF kernel
            process_type: Type of process ('normal' or 'cox')
        """
        self.w = w
        self.gamma = gamma
        self.process_type = process_type

    def compute_mmd_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute MMD-based change point detection scores
        
        Parameters:
            X: Time series data
            
        Returns:
            scores: MMD scores for each time point
        """
        T = len(X)
        scores = np.zeros(T)
        
        for t in range(self.w, T - self.w):
            L = X[t - self.w:t]
            R = X[t:t + self.w]
            
            if self.process_type == 'cox':
                # Apply log transformation for Cox process
                L = np.log(L + 1e-10)
                R = np.log(R + 1e-10)
            
            K_LL = rbf_kernel(L.reshape(-1, 1), L.reshape(-1, 1), gamma=self.gamma)
            K_RR = rbf_kernel(R.reshape(-1, 1), R.reshape(-1, 1), gamma=self.gamma)
            K_LR = rbf_kernel(L.reshape(-1, 1), R.reshape(-1, 1), gamma=self.gamma)
            # Maximum Mean Discrepancy score 
            mmd = (np.sum(K_LL) - np.trace(K_LL)) / (self.w * (self.w - 1))
            mmd += (np.sum(K_RR) - np.trace(K_RR)) / (self.w * (self.w - 1))
            mmd -= 2 * np.mean(K_LR)
            
            scores[t] = mmd
        
        return scores

    def detect_change_points(self, scores: np.ndarray, 
                           threshold: Optional[float] = None,
                           min_distance: int = 50) -> List[int]:
        """
        Detect change points in the time series
        
        Parameters:
            scores: MMD scores
            threshold: Detection threshold
            min_distance: Minimum distance between change points
            
        Returns:
            change_points: List of detected change point locations
        """
        threshold = np.mean(scores) + 2 * np.std(scores)
        # threshold = np.percentile(scores, 95)
        if threshold is not None:
            change_points = np.where(scores > threshold)[0]
        else:
            change_points = []
            for i in range(1, len(scores) - 1):
                if scores[i] > scores[i-1] and scores[i] > scores[i+1]:
                    change_points.append(i)
            change_points = np.array(change_points)
        
        if len(change_points) > 1:
            filtered_points = [change_points[0]]
            for point in change_points[1:]:
                if point - filtered_points[-1] >= min_distance:
                    filtered_points.append(point)
            change_points = filtered_points
        
        return change_points

    def analyze_distributions(self, X: np.ndarray, change_points: List[int]) -> Dict:
        """
        Analyze distributions before and after change points
        
        Parameters:
            X: Time series data
            change_points: Detected change point locations
            
        Returns:
            distributions: Dictionary containing distribution parameters
        """
        distributions = {}
        
        for i, cp in enumerate(change_points):
            before_data = X[max(0, cp-self.w):cp]
            after_data = X[cp:min(len(X), cp+self.w)]
            
            if self.process_type == 'cox':
                # Use Poisson distribution for Cox process
                before_lambda = np.mean(before_data)
                after_lambda = np.mean(after_data)
                
                distributions[cp] = {
                    'before': {
                        'lambda': before_lambda,
                        'data': before_data
                    },
                    'after': {
                        'lambda': after_lambda,
                        'data': after_data
                    }
                }
            else:
                # Use normal distribution for normal process
                before_params = stats.norm.fit(before_data)
                after_params = stats.norm.fit(after_data)
                
                distributions[cp] = {
                    'before': {
                        'mean': before_params[0],
                        'std': before_params[1],
                        'data': before_data
                    },
                    'after': {
                        'mean': after_params[0],
                        'std': after_params[1],
                        'data': after_data
                    }
                }
        
        return distributions

    def predict_future(self, X: np.ndarray, 
                      change_points: List[int],
                      distributions: Dict,
                      steps: int = 50) -> np.ndarray:
        """
        Predict future data based on current distribution
        
        Parameters:
            X: Time series data
            change_points: Detected change point locations
            distributions: Distribution parameters
            steps: Number of steps to predict
            
        Returns:
            future_data: Predicted future values
        """
        if not change_points:
            last_data = X[-self.w:]
            if self.process_type == 'cox':
                last_lambda = np.mean(last_data)
                return np.random.poisson(last_lambda, steps)
            else:
                mean, std = stats.norm.fit(last_data)
                return np.random.normal(mean, std, steps)
        
        last_cp = change_points[-1]
        last_dist = distributions[last_cp]['after']
        
        if self.process_type == 'cox':
            return np.random.poisson(last_dist['lambda'], steps)
        else:
            return np.random.normal(last_dist['mean'], last_dist['std'], steps)

    def plot_analysis(self, X: np.ndarray, 
                     scores: np.ndarray,
                     change_points: List[int],
                     distributions: Dict,
                     future_data: Optional[np.ndarray] = None,
                     title: str = "Change Point Analysis") -> None:
        """
        Visualize the analysis results
        
        Parameters:
            X: Original time series
            scores: MMD scores
            change_points: Detected change point locations
            distributions: Distribution parameters
            future_data: Predicted future values
            title: Plot title
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), height_ratios=[2, 1, 1])
        
        # Plot original data and change points
        ax1.plot(X, label='Original Data')
        for cp in change_points:
            ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.5)
        if future_data is not None:
            future_x = np.arange(len(X), len(X) + len(future_data))
            ax1.plot(future_x, future_data, 'g--', label='Predicted Data')
        ax1.set_title(title)
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot MMD scores
        ax2.plot(scores, label='MMD Score')
        ax2.set_ylabel('MMD Score')
        ax2.grid(True)
        
        # Plot distributions
        if change_points:
            last_cp = change_points[-1]
            before_data = distributions[last_cp]['before']['data']
            after_data = distributions[last_cp]['after']['data']
            
            if self.process_type == 'cox':
                # Use histograms for Cox process
                ax3.hist(before_data, alpha=0.5, label='Before Change', bins=30)
                ax3.hist(after_data, alpha=0.5, label='After Change', bins=30)
            else:
                # Use PDF for normal process
                x = np.linspace(min(min(before_data), min(after_data)),
                              max(max(before_data), max(after_data)), 100)
                before_pdf = stats.norm.pdf(x, *stats.norm.fit(before_data))
                after_pdf = stats.norm.pdf(x, *stats.norm.fit(after_data))
                ax3.plot(x, before_pdf, label='Before Change')
                ax3.plot(x, after_pdf, label='After Change')
            
            ax3.set_xlabel('Value')
            ax3.set_ylabel('Frequency' if self.process_type == 'cox' else 'Density')
            ax3.legend()
            ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

def kernel_cpd(X: np.ndarray,
              w: int = 50,
              gamma: float = 1.0,
              threshold: Optional[float] = None,
              min_distance: int = 50,
              plot: bool = True,
              predict_steps: int = 50,
              process_type: str = 'normal') -> Tuple[np.ndarray, List[int], Dict, np.ndarray]:
    """
    Complete kernel-based change point detection pipeline
    
    Parameters:
        X: Time series data
        w: Window size
        gamma: Parameter for RBF kernel
        threshold: Detection threshold
        min_distance: Minimum distance between change points
        plot: Whether to plot results
        predict_steps: Number of steps to predict
        process_type: Type of process ('normal' or 'cox')
    
    Returns:
        scores: MMD score sequence
        change_points: Detected change point locations
        distributions: Distribution parameters
        future_data: Predicted future values
    """
    detector = KernelCPD(w, gamma, process_type)
    scores = detector.compute_mmd_score(X)
    change_points = detector.detect_change_points(scores, threshold, min_distance)
    distributions = detector.analyze_distributions(X, change_points)
    future_data = detector.predict_future(X, change_points, distributions, predict_steps)
    
    if plot:
        detector.plot_analysis(X, scores, change_points, distributions, future_data)
    
    return scores, change_points, distributions, future_data

if __name__ == "__main__":
    # Read data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    
    # Test normal process data
    data5 = pd.read_csv(os.path.join(data_dir, 'data5.csv')).values.flatten()
    print("\nAnalyzing Normal Process Data (Z(0)=5):")
    scores5, cps5, dists5, future5 = kernel_cpd(
        data5,
        w=10,
        gamma=0.1,
        predict_steps=100,
        process_type='normal'
    )

    data5 = pd.read_csv(os.path.join(data_dir, 'data80.csv')).values.flatten()
    print("\nAnalyzing Normal Process Data (Z(0)=80):")
    scores5, cps5, dists5, future5 = kernel_cpd(
        data5,
        w=10,
        gamma=0.1,
        predict_steps=100,
        process_type='normal'
    )