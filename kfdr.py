import numpy as np
from typing import Tuple, List, Optional
from kernel_cpd import compute_mmd_score, detect_change_points
import matplotlib.pyplot as plt

def predict_next_change(X: np.ndarray,
                       w: int = 90,
                       gamma: float = 1.0,
                       lookback: int = 100,
                       threshold: float = 0.1) -> Tuple[float, float]:
    """
    Predict the likelihood of a change point in the next time slot
    
    Parameters:
        X: Time series data
        w: Window size for MMD computation
        gamma: Parameter for RBF kernel
        lookback: Number of past time points to consider
        threshold: Threshold for change point detection
    
    Returns:
        prediction_score: Predicted MMD score for next time slot
        probability: Probability of change point occurrence
    """
    # Get the most recent data points
    recent_data = X[-lookback:]
    
    # Compute MMD scores for the recent data
    scores = compute_mmd_score(recent_data, w, gamma)
    
    # Use the last computed MMD score as prediction
    prediction_score = scores[-1]
    
    # Convert score to probability using sigmoid function
    probability = 1 / (1 + np.exp(-prediction_score))
    
    return prediction_score, probability

def validate_prediction(X: np.ndarray,
                      true_change_points: List[int],
                      w: int = 50,
                      gamma: float = 1.0,
                      lookback: int = 100) -> Tuple[float, float, List[float]]:
    """
    Validate the prediction performance using historical data
    
    Parameters:
        X: Time series data
        true_change_points: List of actual change point locations
        w: Window size for MMD computation
        gamma: Parameter for RBF kernel
        lookback: Number of past time points to consider
    
    Returns:
        accuracy: Prediction accuracy
        f1_score: F1 score for change point detection
        prediction_scores: List of prediction scores
    """
    prediction_scores = []
    true_labels = []
    
    # Generate prediction scores for each time point
    for t in range(lookback, len(X)):
        # Get data up to current time point
        current_data = X[:t]
        
        # Predict next change point
        score, _ = predict_next_change(current_data, w, gamma, lookback)
        prediction_scores.append(score)
        
        # Check if next time point is a change point
        is_change = any(abs(t - cp) <= w for cp in true_change_points)
        true_labels.append(1 if is_change else 0)
    
    # Convert scores to binary predictions using threshold
    threshold = np.median(prediction_scores)
    predictions = [1 if score > threshold else 0 for score in prediction_scores]
    
    # Compute accuracy and F1 score
    accuracy = np.mean([p == t for p, t in zip(predictions, true_labels)])
    
    # Compute F1 score
    true_positives = sum([p == 1 and t == 1 for p, t in zip(predictions, true_labels)])
    false_positives = sum([p == 1 and t == 0 for p, t in zip(predictions, true_labels)])
    false_negatives = sum([p == 0 and t == 1 for p, t in zip(predictions, true_labels)])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, f1_score, prediction_scores

def plot_validation_results(X: np.ndarray,
                          prediction_scores: List[float],
                          true_change_points: List[int],
                          title: str = "Change Point Prediction Validation") -> None:
    """
    Visualize prediction validation results
    
    Parameters:
        X: Original time series
        prediction_scores: List of prediction scores
        true_change_points: List of actual change point locations
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Plot original data with true change points
    ax1.plot(X, label='Original Data')
    for cp in true_change_points:
        ax1.axvline(x=cp, color='r', linestyle='--', alpha=0.5, label='True Change Point')
    ax1.set_title(title)
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # Plot prediction scores
    ax2.plot(range(len(prediction_scores)), prediction_scores, label='Prediction Score')
    ax2.set_ylabel('Prediction Score')
    ax2.set_xlabel('Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Read data
    data5 = pd.read_csv('/Users/qianxinhui/Desktop/NU-Research/change-point-detection/data/data5.csv').values.flatten()
    data80 = pd.read_csv('/Users/qianxinhui/Desktop/NU-Research/change-point-detection/data/data80.csv').values.flatten()
    # First, detect change points in the data
    from kernel_cpd import kernel_cpd
    scores, change_points = kernel_cpd(data5, w=50, gamma=0.1, plot=False)
    
    # Validate prediction performance
    accuracy, f1_score, pred_scores = validate_prediction(
        data5, 
        change_points,
        w=50,
        gamma=0.1,
        lookback=100
    )
    
    print(f"Prediction Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    
    # Plot validation results
    plot_validation_results(data5, pred_scores, change_points)
    
    # Predict next change point
    next_score, probability = predict_next_change(data5)
    print(f"\nNext time slot prediction:")
    print(f"MMD Score: {next_score:.4f}")
    print(f"Probability of change: {probability:.2%}")