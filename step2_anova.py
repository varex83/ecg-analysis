import numpy as np
import pandas as pd
from scipy import stats

def perform_one_way_anova(data):
    """Perform one-way ANOVA analysis for 12 channels."""
    k = data.shape[1]  # number of channels = 12
    n = data.shape[0]  # number of samples per channel
    
    channel_vars = np.var(data, axis=0, ddof=1)
    S0_squared = np.mean(channel_vars)
    overall_mean = np.mean(data)
    S_squared = np.var(data.flatten(), ddof=1)
    channel_means = np.mean(data, axis=0)
    SA_squared = (n / (k - 1)) * np.sum((channel_means - overall_mean) ** 2)
    
    f_statistic = SA_squared / S0_squared
    df1 = k - 1
    df2 = k * (n - 1)
    p_value = 1 - stats.f.cdf(f_statistic, df1, df2)
    f_critical = stats.f.ppf(0.95, df1, df2)
    
    return {
        'channel_variances': channel_vars,
        'pooled_variance': S0_squared,
        'overall_variance': S_squared,
        'factor_variance': SA_squared,
        'f_statistic': f_statistic,
        'f_critical': f_critical,
        'p_value': p_value,
        'df1': df1, 'df2': df2,
        'channel_means': channel_means,
        'overall_mean': overall_mean
    }

def save_anova_results(results, filename, folder):
    """Save ANOVA results to files."""
    # Save detailed results
    anova_df = pd.DataFrame({
        'Channel': [f'Channel {i+1}' for i in range(12)],
        'Mean': results['channel_means'],
        'Variance': results['channel_variances']
    })
    anova_df.to_csv(f"{folder}/{filename}_anova_results.csv", index=False)
    
    # Save summary
    with open(f"{folder}/{filename}_anova_summary.txt", 'w') as f:
        f.write(f"One-Way ANOVA Results for {filename}\n")
        f.write("-" * 50 + "\n")
        f.write(f"F-statistic: {results['f_statistic']:.4f}\n")
        f.write(f"F-critical (α=0.05): {results['f_critical']:.4f}\n")
        f.write(f"p-value: {results['p_value']:.4e}\n")
        f.write(f"Degrees of freedom: ({results['df1']}, {results['df2']})\n") 