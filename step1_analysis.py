import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

def load_cardiogram_data(file_path):
    """
    Load cardiogram data from file where each line contains 12 comma-separated values
    representing measurements from 12 channels.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            channels = [float(x.strip()) for x in line.split(',')]
            if len(channels) != 12:
                raise ValueError(f"Expected 12 channels, but got {len(channels)} values")
            data.append(channels)
    
    data = np.array(data)
    print(f"Loaded data shape: {data.shape}")
    return data

def mean_absolute_difference(data):
    """Calculate mean absolute difference (Gini mean difference)"""
    n = len(data)
    abs_diff_sum = sum(abs(x - y) for i, x in enumerate(data) 
                      for y in data[i + 1:])
    return (2 * abs_diff_sum) / (n * (n - 1)) if n > 1 else 0

def calculate_statistics(data):
    """Calculate various statistical parameters for each channel."""
    stats_dict = {
        'Mean': np.mean(data, axis=0),
        'Harmonic Mean': stats.hmean(data - np.min(data) + 1, axis=0),
        'Geometric Mean': stats.gmean(data - np.min(data) + 1, axis=0),
        'Variance': np.var(data, axis=0),
        'Median': np.median(data, axis=0),
        'Mode': stats.mode(data, axis=0).mode[0],
        'Skewness': stats.skew(data, axis=0),
        'Kurtosis': stats.kurtosis(data, axis=0),
        'Gini Mean Difference': np.array([mean_absolute_difference(data[:, i]) 
                                        for i in range(data.shape[1])])
    }
    return pd.DataFrame(stats_dict, index=[f'Channel {i+1}' for i in range(data.shape[1])])

def plot_channels(data, sampling_rate=500, filename=''):
    """Plot all 12 channels of cardiogram data."""
    time = np.arange(len(data)) / sampling_rate
    fig, axes = plt.subplots(6, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i in range(12):
        axes[i].plot(time, data[:, i])
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_xlabel('Time (seconds)')
        axes[i].set_ylabel('Amplitude')
    
    plt.suptitle(f'Cardiogram Channels - {filename}')
    plt.tight_layout()
    return fig

def plot_histograms(data, filename=''):
    """Plot histograms for all channels."""
    fig, axes = plt.subplots(6, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for i in range(12):
        sns.histplot(data[:, i], kde=True, ax=axes[i])
        axes[i].set_title(f'Channel {i+1} Distribution')
        
        stat, p_value = stats.shapiro(data[:, i])
        axes[i].text(0.05, 0.95, f'Shapiro-Wilk p-value: {p_value:.2e}',
                    transform=axes[i].transAxes)
    
    plt.suptitle(f'Channel Distributions - {filename}')
    plt.tight_layout()
    return fig

def normalize_data(data):
    """Normalize data to zero mean and unit variance."""
    return stats.zscore(data, axis=0)

def perform_initial_analysis(data, filename, output_folder):
    """Perform initial analysis including loading, statistics, and visualization."""
    # Calculate statistics
    stats_df = calculate_statistics(data)
    stats_df.to_csv(f"{output_folder}/{filename}_statistics.csv")
    
    # Normalize data
    normalized_data = normalize_data(data)
    norm_stats = pd.DataFrame({
        'Mean': np.mean(normalized_data, axis=0),
        'Std': np.std(normalized_data, axis=0)
    }, index=[f'Channel {i+1}' for i in range(data.shape[1])])
    norm_stats.to_csv(f"{output_folder}/{filename}_normalized_stats.csv")
    
    # Create plots
    channel_plot = plot_channels(data, filename=filename)
    channel_plot.savefig(f"{output_folder}/{filename}_channels.png")
    
    hist_plot = plot_histograms(data, filename=filename)
    hist_plot.savefig(f"{output_folder}/{filename}_distributions.png")
    
    return {
        'statistics': stats_df,
        'normalized_data': normalized_data,
        'normalization_check': norm_stats
    } 