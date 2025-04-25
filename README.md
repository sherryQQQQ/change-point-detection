# Change Point Detection

This project focuses on implementing and analyzing change point detection algorithms.

## Setup

1. Clone the repository
2. Install dependencies
3. Run the analysis

## Features
- Summary of Change Point Detection Analysis
- We analyzed two datasets (data_5 and data_80) using a window size of 4 for both.
- We generated predictions for both datasets and saved the results to CSV files.
- We calculated performance metrics including MSE and standard deviation for both datasets.
- We visualized the data with and without change points, as well as the deviation patterns.
The analysis helps us understand how well our prediction model performs and where potential change points occur in the time series data.

## Latest Updates (v2) 2025-04-25

### Kernel-based Change Point Detection

The latest version (v2) introduces a kernel-based Maximum Mean Discrepancy (MMD) approach for change point detection with the following enhancements:

- Using the past the data to simulate data in the next interval
- 
### Usage Example
Data: Cox Process Simulation data with different initial values (5,80) and different breaks M (500,5000).