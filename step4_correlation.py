import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def normalize_data_matrix(data):
    """
    Normalize all variables according to:
    x_ij* = (x_ij - mean_j) / std_j
    """
    return stats.zscore(data, axis=0)

def compute_correlation_matrix(normalized_data):
    """
    Compute correlation matrix using normalized data:
    r = (1/n) * (X* · X*)
    """
    return np.corrcoef(normalized_data.T)

def find_highly_correlated_groups(corr_matrix, threshold=0.7):
    """
    Find groups of 3-4 parameters with high correlation (above threshold).
    Returns multiple groups sorted by average correlation strength.
    """
    n = len(corr_matrix)
    groups = []
    
    # Try all possible combinations of 3 and 4 parameters
    for size in [3, 4]:
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    indices = [i, j, k]
                    if size == 4:
                        for l in range(k + 1, n):
                            indices_4 = indices + [l]
                            sub_matrix = corr_matrix[np.ix_(indices_4, indices_4)]
                            # Check if all correlations are above threshold
                            if np.all(np.abs(sub_matrix[np.triu_indices(size, 1)]) > threshold):
                                avg_corr = np.mean(np.abs(sub_matrix[np.triu_indices(size, 1)]))
                                groups.append({
                                    'indices': indices_4,
                                    'size': size,
                                    'avg_correlation': avg_corr
                                })
                    else:
                        sub_matrix = corr_matrix[np.ix_(indices, indices)]
                        if np.all(np.abs(sub_matrix[np.triu_indices(size, 1)]) > threshold):
                            avg_corr = np.mean(np.abs(sub_matrix[np.triu_indices(size, 1)]))
                            groups.append({
                                'indices': indices,
                                'size': size,
                                'avg_correlation': avg_corr
                            })
    
    # Sort groups by average correlation strength
    return sorted(groups, key=lambda x: x['avg_correlation'], reverse=True)

def partial_correlation_2var(corr_matrix, a, b, c):
    """Calculate partial correlation between a and b without c."""
    r_ab = corr_matrix[a, b]
    r_ac = corr_matrix[a, c]
    r_bc = corr_matrix[b, c]
    
    numerator = r_ab - r_ac * r_bc
    denominator = np.sqrt((1 - r_ac**2) * (1 - r_bc**2))
    
    return numerator / denominator

def partial_correlation_3var(corr_matrix, a, b, c, d):
    """Calculate partial correlation between a and b without c and d."""
    r_ab = corr_matrix[a, b]
    r_ac = corr_matrix[a, c]
    r_ad = corr_matrix[a, d]
    r_bc = corr_matrix[b, c]
    r_bd = corr_matrix[b, d]
    r_cd = corr_matrix[c, d]
    
    # This is a simplified version - for exact calculation you might want to use
    # more sophisticated methods like inverse correlation matrix
    r_ab_c = partial_correlation_2var(corr_matrix, a, b, c)
    r_ab_d = partial_correlation_2var(corr_matrix, a, b, d)
    
    return (r_ab_c + r_ab_d) / 2

def multiple_correlation_2factors(corr_matrix, a, b, c):
    """Calculate multiple correlation coefficient for a based on b and c."""
    r_ab = corr_matrix[a, b]
    r_ac = corr_matrix[a, c]
    r_bc = corr_matrix[b, c]
    
    numerator = r_ab**2 + r_ac**2 - 2*r_ab*r_ac*r_bc
    denominator = 1 - r_bc**2
    
    return np.sqrt(numerator / denominator)

def multiple_correlation_3factors(corr_matrix, a, b, c, d):
    """Calculate multiple correlation coefficient for a based on b, c, and d."""
    r_ab = corr_matrix[a, b]
    r_ac_b = partial_correlation_2var(corr_matrix, a, c, b)
    r_ad_bc = partial_correlation_3var(corr_matrix, a, d, b, c)
    
    return np.sqrt(1 - (1 - r_ab**2)*(1 - r_ac_b**2)*(1 - r_ad_bc**2))

def plot_correlation_matrix(corr_matrix, filename):
    """Plot correlation matrix as a heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()

def perform_correlation_analysis(data):
    """Perform complete correlation analysis."""
    # Step 1: Normalize data
    normalized_data = normalize_data_matrix(data)
    
    # Step 2: Compute correlation matrix
    corr_matrix = compute_correlation_matrix(normalized_data)
    
    # Step 3: Find highly correlated groups
    groups = find_highly_correlated_groups(corr_matrix)
    
    # Select the best group for further analysis
    best_group = groups[0]['indices']
    
    # Steps 4-7: Calculate various correlation coefficients
    results = {
        'correlation_matrix': corr_matrix,
        'highly_correlated_groups': groups,
        'partial_correlations': {
            'r_ab_c': partial_correlation_2var(corr_matrix, best_group[0], best_group[1], best_group[2]),
            'r_ac_b': partial_correlation_2var(corr_matrix, best_group[0], best_group[2], best_group[1]),
        },
        'multiple_correlations': {
            'r_a_bc': multiple_correlation_2factors(corr_matrix, best_group[0], best_group[1], best_group[2]),
        }
    }
    
    if len(best_group) == 4:
        results['partial_correlations'].update({
            'r_ab_cd': partial_correlation_3var(corr_matrix, best_group[0], best_group[1], best_group[2], best_group[3]),
            'r_ad_bc': partial_correlation_3var(corr_matrix, best_group[0], best_group[3], best_group[1], best_group[2])
        })
        results['multiple_correlations']['R_a_bcd'] = multiple_correlation_3factors(
            corr_matrix, best_group[0], best_group[1], best_group[2], best_group[3]
        )
    
    return results

def save_correlation_results(results, filename, folder):
    """Save correlation analysis results."""
    # Save correlation matrix
    pd.DataFrame(
        results['correlation_matrix'],
        columns=[f'Channel {i+1}' for i in range(len(results['correlation_matrix']))],
        index=[f'Channel {i+1}' for i in range(len(results['correlation_matrix']))]
    ).to_csv(f"{folder}/{filename}_correlation_matrix.csv")
    
    # Save correlation plot
    plot_correlation_matrix(results['correlation_matrix'], filename).savefig(
        f"{folder}/{filename}_correlation_matrix.png"
    )
    
    # Save detailed results
    with open(f"{folder}/{filename}_correlation_analysis.txt", 'w') as f:
        f.write("Correlation Analysis Results\n")
        f.write("-" * 50 + "\n\n")
        
        # Write highly correlated groups
        f.write("Highly Correlated Groups:\n")
        for i, group in enumerate(results['highly_correlated_groups'][:5], 1):
            channels = [f"Channel {idx+1}" for idx in group['indices']]
            f.write(f"Group {i}: {', '.join(channels)}\n")
            f.write(f"Average correlation: {group['avg_correlation']:.4f}\n\n")
        
        # Write partial correlations
        f.write("\nPartial Correlations:\n")
        for name, value in results['partial_correlations'].items():
            f.write(f"{name}: {value:.4f}\n")
        
        # Write multiple correlations
        f.write("\nMultiple Correlations:\n")
        for name, value in results['multiple_correlations'].items():
            f.write(f"{name}: {value:.4f}\n") 