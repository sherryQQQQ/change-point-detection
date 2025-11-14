# predictor/base.py
"""Base class for all rolling predictors"""

import abc

class RollingPredictor(abc.ABC):
    """
    Abstract base class for rolling prediction algorithms.
    
    All predictor implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    @abc.abstractmethod
    def rolling_prediction(self, times, values, verbose=True):
        """
        Execute rolling prediction: for each time point, use previous data 
        to predict the value at that point.
        
        Args:
            times: Complete time series
            values: Complete observation sequence
            verbose: Whether to output detailed information
            
        Returns:
            dict: Summary of prediction results
        """
        pass
    
    @abc.abstractmethod
    def get_summary_metrics(self):
        """
        Calculate prediction performance metrics.
        
        Returns:
            dict: Dictionary containing performance metrics
        """
        pass
    
    def plot_rolling_predictions(self, original_times=None, original_values=None, 
                                 show_confidence=True, show_errors=True):
        """
        Plot rolling prediction results.
        Optional method - subclasses can override if needed.
        
        Args:
            original_times: Original complete time series (for context)
            original_values: Original complete observation sequence
            show_confidence: Whether to show confidence interval
            show_errors: Whether to show error plot
        """
        print("plot_rolling_predictions not implemented for this predictor")
        pass