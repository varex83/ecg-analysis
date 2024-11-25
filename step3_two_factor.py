import numpy as np
from scipy import stats
import pandas as pd

def create_two_factor_table(data, n_parts=5):
    """
    Create a two-factor table where:
    - Factor A: Channels (12 levels)
    - Factor B: Time segments (5 levels)
    - Each cell contains 1000 data points
    
    Returns:
    - table: Dictionary with shape (5, 12) containing arrays of 1000 points each
    - means: Array of shape (5, 12) containing mean values for each cell
    """
    k = data.shape[1]  # number of channels (A levels)
    m = n_parts        # number of time segments (B levels)
    n = len(data) // m # points per segment
    
    # Reshape data into (B_levels, points_per_segment, A_levels)
    table = {}
    means = np.zeros((m, k))
    
    for i in range(m):  # B levels (time segments)
        start_idx = i * n
        end_idx = (i + 1) * n
        for j in range(k):  # A levels (channels)
            table[(i, j)] = data[start_idx:end_idx, j]
            means[i, j] = np.mean(table[(i, j)])
    
    return table, means

def calculate_summary_indicators(means):
    """Calculate Q1, Q2, Q3, Q4 for independent factors case."""
    k = means.shape[1]  # number of channels (A levels)
    m = means.shape[0]  # number of time segments (B levels)
    
    # Q1 = sum of squared means
    Q1 = np.sum(means ** 2)
    
    # Calculate sums along columns (Xi) and rows (Xj)
    Xi = np.sum(means, axis=0)  # sum along B (time segments)
    Xj = np.sum(means, axis=1)  # sum along A (channels)
    
    # Q2 = (1/m) * sum(Xi^2)
    Q2 = np.sum(Xi ** 2) / m
    
    # Q3 = (1/k) * sum(Xj^2)
    Q3 = np.sum(Xj ** 2) / k
    
    # Q4 = (1/mk) * (sum(Xi))^2 = (1/mk) * (sum(Xj))^2
    Q4 = (np.sum(Xi) ** 2) / (m * k)
    
    return {
        'Q1': Q1,
        'Q2': Q2,
        'Q3': Q3,
        'Q4': Q4,
        'Xi': Xi,
        'Xj': Xj
    }

def calculate_dependent_factors(table):
    """Calculate Q5 for dependent factors case."""
    Q5 = 0
    for values in table.values():
        Q5 += np.sum(values ** 2)
    return Q5

def perform_two_factor_analysis(data):
    """Perform complete two-factor analysis."""
    k = data.shape[1]  # number of channels (A levels)
    m = 5              # number of time segments (B levels)
    
    # Create two-factor table
    table, means = create_two_factor_table(data)
    n = len(table[(0, 0)])  # points per cell
    
    # Calculate summary indicators
    indicators = calculate_summary_indicators(means)
    Q5 = calculate_dependent_factors(table)
    
    # Calculate variances
    S0_squared = (indicators['Q1'] + indicators['Q4'] - 
                 indicators['Q2'] - indicators['Q3']) / (k * m - 1)
    SA_squared = (indicators['Q2'] - indicators['Q4']) / (k - 1)
    SB_squared = (indicators['Q3'] - indicators['Q4']) / (m - 1)
    SAB_squared = (Q5 - n * indicators['Q1']) / (m * k * (n - 1))
    
    # Calculate F-statistics and p-values
    f1_A = k - 1
    f1_B = m - 1
    f2 = (k - 1) * (m - 1)
    f2_AB = m * k * (n - 1)
    
    F_A = SA_squared / S0_squared
    F_B = SB_squared / S0_squared
    F_AB = SAB_squared / S0_squared
    
    F_crit_A = stats.f.ppf(0.95, f1_A, f2)
    F_crit_B = stats.f.ppf(0.95, f1_B, f2)
    F_crit_AB = stats.f.ppf(0.95, f2, f2_AB)
    
    p_value_A = 1 - stats.f.cdf(F_A, f1_A, f2)
    p_value_B = 1 - stats.f.cdf(F_B, f1_B, f2)
    p_value_AB = 1 - stats.f.cdf(F_AB, f2, f2_AB)
    
    return {
        'means_table': means,
        'variances': {
            'S0_squared': S0_squared,
            'SA_squared': SA_squared,
            'SB_squared': SB_squared,
            'SAB_squared': SAB_squared
        },
        'F_statistics': {
            'F_A': F_A,
            'F_B': F_B,
            'F_AB': F_AB,
            'F_crit_A': F_crit_A,
            'F_crit_B': F_crit_B,
            'F_crit_AB': F_crit_AB
        },
        'p_values': {
            'p_value_A': p_value_A,
            'p_value_B': p_value_B,
            'p_value_AB': p_value_AB
        },
        'summary_indicators': indicators
    }

def save_two_factor_results(results, filename, folder):
    """Save two-factor analysis results."""
    # Save means table
    means_df = pd.DataFrame(
        results['means_table'],
        index=[f'B{i+1}' for i in range(5)],
        columns=[f'A{i+1}' for i in range(12)]
    )
    means_df.to_csv(f"{folder}/{filename}_two_factor_means.csv")
    
    # Save analysis summary
    with open(f"{folder}/{filename}_two_factor_summary.txt", 'w') as f:
        f.write("Two-Factor Analysis Results\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("Variances:\n")
        for name, value in results['variances'].items():
            f.write(f"{name}: {value:.4f}\n")
        
        f.write("\nF-statistics:\n")
        for name, value in results['F_statistics'].items():
            f.write(f"{name}: {value:.4f}\n")
        
        f.write("\np-values:\n")
        for name, value in results['p_values'].items():
            f.write(f"{name}: {value:.4e}\n")
        
        # Add conclusions
        f.write("\nConclusions:\n")
        f.write("Factor A (Channels): ")
        if results['F_statistics']['F_A'] > results['F_statistics']['F_crit_A']:
            f.write("Significant\n")
        else:
            f.write("Not significant\n")
            
        f.write("Factor B (Time segments): ")
        if results['F_statistics']['F_B'] > results['F_statistics']['F_crit_B']:
            f.write("Significant\n")
        else:
            f.write("Not significant\n")
            
        f.write("Interaction A×B: ")
        if results['F_statistics']['F_AB'] > results['F_statistics']['F_crit_AB']:
            f.write("Significant\n")
        else:
            f.write("Not significant\n") 