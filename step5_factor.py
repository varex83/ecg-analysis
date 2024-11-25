import numpy as np
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_eigenvalues(corr_matrix):
    """
    Compute eigenvalues of correlation matrix and sort them in descending order.
    Returns eigenvalues and explained variance ratios.
    """
    eigenvals, eigenvecs = linalg.eigh(corr_matrix)
    # Sort in descending order
    idx = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Calculate explained variance ratios
    total_var = np.sum(eigenvals)
    explained_var = eigenvals / total_var * 100
    cumulative_var = np.cumsum(explained_var)
    
    return {
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs,
        'explained_variance': explained_var,
        'cumulative_variance': cumulative_var
    }

def verify_eigenvector_conditions(eigenvectors):
    """
    Verify orthogonality and normalization of eigenvectors.
    Returns verification matrix where diagonal should be 1 and off-diagonal close to 0.
    """
    return np.dot(eigenvectors.T, eigenvectors)

def compute_principal_components(data, eigenvectors):
    """Compute principal components from data and eigenvectors."""
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    # Project data onto eigenvectors
    return np.dot(centered_data, eigenvectors)

def verify_pc_properties(principal_components, eigenvalues, n_samples):
    """
    Verify properties of principal components:
    1. Sum of each PC should be close to 0
    2. Variance of each PC should equal corresponding eigenvalue
    3. PCs should be uncorrelated
    """
    pc_sums = np.sum(principal_components, axis=0)
    pc_vars = np.var(principal_components, axis=0) * (n_samples - 1) / n_samples
    pc_corr = np.corrcoef(principal_components.T)
    
    return {
        'pc_sums': pc_sums,
        'pc_variances': pc_vars,
        'pc_correlations': pc_corr,
        'eigenvalues': eigenvalues
    }

def plot_explained_variance(explained_var, cumulative_var):
    """Plot explained variance ratio and cumulative variance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scree plot
    ax1.plot(range(1, len(explained_var) + 1), explained_var, 'bo-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio (%)')
    ax1.set_title('Scree Plot')
    
    # Cumulative variance plot
    ax2.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance (%)')
    ax2.set_title('Cumulative Explained Variance')
    ax2.axhline(y=97, color='g', linestyle='--', label='97% threshold')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def perform_factor_analysis(data):
    """Perform complete factor analysis."""
    # Compute correlation matrix
    corr_matrix = np.corrcoef(data.T)
    
    # Step 1 & 2: Compute and sort eigenvalues
    eigen_results = compute_eigenvalues(corr_matrix)
    
    # Step 3: Verify eigenvector conditions
    verification_matrix = verify_eigenvector_conditions(eigen_results['eigenvectors'])
    
    # Step 4: Compute principal components
    principal_components = compute_principal_components(data, eigen_results['eigenvectors'])
    
    # Step 5: Verify PC properties
    pc_verification = verify_pc_properties(
        principal_components, 
        eigen_results['eigenvalues'],
        len(data)
    )
    
    return {
        'eigenvalues': eigen_results['eigenvalues'],
        'eigenvectors': eigen_results['eigenvectors'],
        'explained_variance': eigen_results['explained_variance'],
        'cumulative_variance': eigen_results['cumulative_variance'],
        'verification_matrix': verification_matrix,
        'principal_components': principal_components,
        'pc_verification': pc_verification
    }

def plot_principal_components(principal_components, eigenvalues, filename, folder):
    """
    Plot principal components visualization.
    
    Parameters:
    principal_components : array-like
        Principal components matrix
    eigenvalues : array-like
        Eigenvalues for each component
    """
    # Plot first 3 principal components
    fig = plt.figure(figsize=(15, 5))
    
    # Plot PC1
    plt.subplot(131)
    plt.plot(principal_components[:, 0])
    plt.title(f'PC1 (Variance: {eigenvalues[0]:.2f}%)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    # Plot PC2
    plt.subplot(132)
    plt.plot(principal_components[:, 1])
    plt.title(f'PC2 (Variance: {eigenvalues[1]:.2f}%)')
    plt.xlabel('Time')
    
    # Plot PC3
    plt.subplot(133)
    plt.plot(principal_components[:, 2])
    plt.title(f'PC3 (Variance: {eigenvalues[2]:.2f}%)')
    plt.xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(f"{folder}/{filename}_principal_components.png")
    plt.close()
    
    # Plot 2D scatter of PC1 vs PC2
    plt.figure(figsize=(10, 8))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
    plt.xlabel(f'PC1 ({eigenvalues[0]:.2f}%)')
    plt.ylabel(f'PC2 ({eigenvalues[1]:.2f}%)')
    plt.title('PC1 vs PC2 Scatter Plot')
    plt.savefig(f"{folder}/{filename}_pc_scatter.png")
    plt.close()
    
    # Plot 3D scatter of first three PCs
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(principal_components[:, 0], 
                        principal_components[:, 1], 
                        principal_components[:, 2],
                        c=range(len(principal_components)),
                        cmap='viridis')
    ax.set_xlabel(f'PC1 ({eigenvalues[0]:.2f}%)')
    ax.set_ylabel(f'PC2 ({eigenvalues[1]:.2f}%)')
    ax.set_zlabel(f'PC3 ({eigenvalues[2]:.2f}%)')
    plt.colorbar(scatter, label='Time Point')
    plt.title('First Three Principal Components')
    plt.savefig(f"{folder}/{filename}_pc_3d.png")
    plt.close()

def plot_pairwise_components(principal_components, eigenvalues, filename, folder):
    """
    Create 2D plots for each pair of principal components.
    
    Parameters:
    principal_components : array-like
        Principal components matrix
    eigenvalues : array-like
        Eigenvalues (explained variance) for each component
    """
    n_components = min(6, principal_components.shape[1])  # Plot first 6 components
    
    # Create pairwise plots
    for i in range(n_components-1):
        for j in range(i+1, n_components):
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot
            plt.scatter(principal_components[:, i], 
                       principal_components[:, j], 
                       alpha=0.5,
                       c=range(len(principal_components)),
                       cmap='viridis')
            
            # Add labels and title
            plt.xlabel(f'PC{i+1} ({eigenvalues[i]:.2f}%)')
            plt.ylabel(f'PC{j+1} ({eigenvalues[j]:.2f}%)')
            plt.title(f'Principal Components {i+1} vs {j+1}')
            
            # Add colorbar to show time progression
            plt.colorbar(label='Time Point')
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(f"{folder}/{filename}_pc{i+1}_pc{j+1}_scatter.png")
            plt.close()
    
    # Create a summary grid plot
    fig, axes = plt.subplots(n_components, n_components, 
                            figsize=(15, 15))
    
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                axes[i, j].scatter(principal_components[:, j], 
                                 principal_components[:, i],
                                 alpha=0.5,
                                 s=1)
                if i == n_components-1:
                    axes[i, j].set_xlabel(f'PC{j+1}')
                if j == 0:
                    axes[i, j].set_ylabel(f'PC{i+1}')
            else:
                # Plot distribution on diagonal
                axes[i, i].hist(principal_components[:, i], 
                              bins=50,
                              orientation='vertical')
                axes[i, i].set_xlabel(f'PC{i+1}')
            
            axes[i, j].tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{folder}/{filename}_pc_pairwise_grid.png")
    plt.close()

def save_factor_analysis_results(results, filename, folder):
    """Save factor analysis results."""
    # Save eigenvalue analysis
    eigenvalue_df = pd.DataFrame({
        'Eigenvalue': results['eigenvalues'],
        'Explained Variance (%)': results['explained_variance'],
        'Cumulative Variance (%)': results['cumulative_variance']
    })
    eigenvalue_df.index = [f"Component {i+1}" for i in range(len(results['eigenvalues']))]
    eigenvalue_df.to_csv(f"{folder}/{filename}_eigenvalues.csv")
    
    # Save eigenvectors
    eigenvector_df = pd.DataFrame(
        results['eigenvectors'],
        columns=[f"PC{i+1}" for i in range(results['eigenvectors'].shape[1])],
        index=[f"Variable {i+1}" for i in range(results['eigenvectors'].shape[0])]
    )
    eigenvector_df.to_csv(f"{folder}/{filename}_eigenvectors.csv")
    
    # Save plots
    plot_explained_variance(
        results['explained_variance'],
        results['cumulative_variance']
    ).savefig(f"{folder}/{filename}_explained_variance.png")
    
    # Save detailed analysis
    with open(f"{folder}/{filename}_factor_analysis.txt", 'w') as f:
        f.write("Factor Analysis Results\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("Eigenvalue Analysis:\n")
        f.write(eigenvalue_df.to_string())
        f.write("\n\n")
        
        f.write("Eigenvector Verification:\n")
        f.write("Orthogonality check (should be identity matrix):\n")
        f.write(pd.DataFrame(results['verification_matrix']).to_string())
        f.write("\n\n")
        
        f.write("Principal Component Properties:\n")
        f.write(f"PC sums (should be close to 0):\n{results['pc_verification']['pc_sums']}\n")
        f.write(f"PC variances (should equal eigenvalues):\n{results['pc_verification']['pc_variances']}\n")
        
        # Add conclusions
        f.write("\nConclusions:\n")
        n_components_97 = np.sum(results['cumulative_variance'] <= 97)
        f.write(f"Number of components explaining 97% of variance: {n_components_97}\n")
        f.write(f"First component explains {results['explained_variance'][0]:.2f}% of variance\n") 
    
    # Add principal components plotting
    principal_components = results.get('principal_components')
    if principal_components is not None:
        plot_principal_components(
            principal_components,
            results['explained_variance'],
            filename,
            folder
        )
    
    # Add pairwise component plots
    principal_components = results.get('principal_components')
    if principal_components is not None:
        plot_pairwise_components(
            principal_components,
            results['explained_variance'],
            filename,
            folder
        )