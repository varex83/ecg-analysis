import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def perform_kmeans_clustering(data, n_clusters):
    """
    Perform k-means clustering on the data.
    Returns cluster assignments and centroids.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    
    return {
        'labels': cluster_labels,
        'centroids': centroids,
        'inertia': kmeans.inertia_,
        'n_iter': kmeans.n_iter_
    }

def perform_knn_clustering(data, n_neighbors=5):
    """
    Perform KNN clustering using nearest neighbors.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    return {
        'distances': distances,
        'indices': indices
    }

def reduce_dimensions_pca(data, n_components=3):
    """
    Reduce data dimensions using PCA.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(scaled_data)
    
    return {
        'reduced_data': reduced_data,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }

def plot_clusters_2d(data, labels, centroids, title):
    """Plot clustering results in 2D using first two dimensions."""
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidth=3)
    plt.title(title)
    plt.colorbar(scatter)
    return plt.gcf()

def plot_clusters_3d(data, labels, centroids, title):
    """Plot clustering results in 3D using first three dimensions."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis')
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
              c='red', marker='x', s=200, linewidth=3)
    ax.set_title(title)
    plt.colorbar(scatter)
    return fig

def detect_r_peaks(data, cluster_labels, n_clusters):
    """
    Detect R-peaks using clustering results.
    
    Parameters:
    data : array-like
        Original ECG data
    cluster_labels : array-like
        Cluster assignments from k-means
    n_clusters : int
        Number of clusters used
        
    Returns:
    array of R-peak indices
    """
    # Calculate mean amplitude for each cluster
    cluster_means = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_means[i] = np.mean(data[cluster_labels == i])
    
    # Find the cluster with highest mean amplitude (likely R-peaks)
    r_peak_cluster = np.argmax(cluster_means)
    
    # Get indices where cluster label matches R-peak cluster
    potential_r_peaks = np.where(cluster_labels == r_peak_cluster)[0]
    
    # Find local maxima within R-peak cluster points
    r_peaks = []
    for i in range(1, len(potential_r_peaks)-1):
        if (data[potential_r_peaks[i]] > data[potential_r_peaks[i-1]] and 
            data[potential_r_peaks[i]] > data[potential_r_peaks[i+1]]):
            r_peaks.append(potential_r_peaks[i])
    
    return np.array(r_peaks)

def plot_r_peaks(data, r_peaks, filename, folder):
    """
    Plot ECG signal with detected R-peaks.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot original signal
    plt.plot(data, 'b-', label='ECG Signal')
    
    # Plot R-peaks
    plt.plot(r_peaks, data[r_peaks], 'ro', label='R-peaks')
    
    plt.title('ECG Signal with Detected R-peaks')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f"{folder}/{filename}_r_peaks.png")
    plt.close()

def calculate_r_peak_statistics(data, r_peaks):
    """
    Calculate statistics for R-peaks.
    """
    # Calculate R-R intervals
    rr_intervals = np.diff(r_peaks)
    
    # Calculate heart rate (assuming sampling rate of 500 Hz)
    heart_rates = 60 * 500 / rr_intervals
    
    return {
        'n_peaks': len(r_peaks),
        'mean_rr': np.mean(rr_intervals),
        'std_rr': np.std(rr_intervals),
        'mean_hr': np.mean(heart_rates),
        'std_hr': np.std(heart_rates),
        'peak_amplitudes': data[r_peaks]
    }

def perform_clustering_analysis(data):
    """Perform complete clustering analysis."""
    # Part A: Clustering on original data
    results_11 = perform_kmeans_clustering(data, n_clusters=11)
    results_7 = perform_kmeans_clustering(data, n_clusters=7)
    
    # KNN clustering
    knn_results = perform_knn_clustering(data)
    
    # Part B: Clustering on reduced dimensions
    pca_results = reduce_dimensions_pca(data)
    pca_clusters_11 = perform_kmeans_clustering(pca_results['reduced_data'], n_clusters=11)
    pca_clusters_7 = perform_kmeans_clustering(pca_results['reduced_data'], n_clusters=7)
    
    # Detect R-peaks for each channel using k=11 clustering
    r_peaks_results = []
    for channel in range(data.shape[1]):
        r_peaks = detect_r_peaks(
            data[:, channel],
            results_11['labels'],
            n_clusters=11
        )
        r_peaks_results.append({
            'peaks': r_peaks,
            'statistics': calculate_r_peak_statistics(data[:, channel], r_peaks)
        })
    
    return {
        'original_clusters_11': results_11,
        'original_clusters_7': results_7,
        'knn_results': knn_results,
        'pca_results': pca_results,
        'pca_clusters_11': pca_clusters_11,
        'pca_clusters_7': pca_clusters_7,
        'r_peaks': r_peaks_results
    }

def compare_cluster_consistency(original_labels, pca_labels):
    """Compare cluster assignments between original and PCA-reduced data."""
    # Create contingency table
    contingency = pd.crosstab(original_labels, pca_labels)
    
    # Calculate consistency metrics
    total_points = len(original_labels)
    matching_points = sum(contingency.max(axis=1))
    consistency = matching_points / total_points
    
    return {
        'contingency_table': contingency,
        'consistency_score': consistency
    }

def save_clustering_results(results, filename, folder, data):
    """
    Save clustering analysis results.
    
    Parameters:
    results : dict
        Clustering analysis results
    filename : str
        Base filename for saving results
    folder : str
        Output folder path
    data : array-like
        Original ECG data
    """
    # Save cluster assignments
    pd.DataFrame({
        'Cluster_11': results['original_clusters_11']['labels'],
        'Cluster_7': results['original_clusters_7']['labels']
    }).to_csv(f"{folder}/{filename}_clusters.csv")
    
    # Save PCA results
    pd.DataFrame({
        'PC1': results['pca_results']['reduced_data'][:, 0],
        'PC2': results['pca_results']['reduced_data'][:, 1],
        'PC3': results['pca_results']['reduced_data'][:, 2],
        'Cluster_11': results['pca_clusters_11']['labels'],
        'Cluster_7': results['pca_clusters_7']['labels']
    }).to_csv(f"{folder}/{filename}_pca_clusters.csv")
    
    # Save PCA variance explained
    pd.DataFrame({
        'Explained_Variance_Ratio': results['pca_results']['explained_variance_ratio'],
        'Cumulative_Variance': results['pca_results']['cumulative_variance']
    }).to_csv(f"{folder}/{filename}_pca_variance.csv")
    
    # Save plots
    plot_clusters_2d(
        results['pca_results']['reduced_data'][:, :2],
        results['pca_clusters_11']['labels'],
        results['pca_clusters_11']['centroids'][:, :2],
        'Clustering Results (k=11) on First Two Principal Components'
    ).savefig(f"{folder}/{filename}_clusters_11.png")
    
    plot_clusters_2d(
        results['pca_results']['reduced_data'][:, :2],
        results['pca_clusters_7']['labels'],
        results['pca_clusters_7']['centroids'][:, :2],
        'Clustering Results (k=7) on First Two Principal Components'
    ).savefig(f"{folder}/{filename}_clusters_7.png")
    
    # Save analysis summary
    consistency_11 = compare_cluster_consistency(
        results['original_clusters_11']['labels'],
        results['pca_clusters_11']['labels']
    )
    
    consistency_7 = compare_cluster_consistency(
        results['original_clusters_7']['labels'],
        results['pca_clusters_7']['labels']
    )
    
    with open(f"{folder}/{filename}_clustering_analysis.txt", 'w') as f:
        f.write("Clustering Analysis Results\n")
        f.write("-" * 50 + "\n\n")
        
        f.write("PCA Analysis:\n")
        f.write(f"Explained variance ratios: {results['pca_results']['explained_variance_ratio']}\n")
        f.write(f"Cumulative variance: {results['pca_results']['cumulative_variance']}\n\n")
        
        f.write("K-means Clustering (k=11):\n")
        f.write(f"Inertia: {results['original_clusters_11']['inertia']}\n")
        f.write(f"Iterations: {results['original_clusters_11']['n_iter']}\n")
        f.write(f"Consistency with PCA: {consistency_11['consistency_score']:.4f}\n\n")
        
        f.write("K-means Clustering (k=7):\n")
        f.write(f"Inertia: {results['original_clusters_7']['inertia']}\n")
        f.write(f"Iterations: {results['original_clusters_7']['n_iter']}\n")
        f.write(f"Consistency with PCA: {consistency_7['consistency_score']:.4f}\n") 
    
    # Save R-peak results
    for channel in range(len(results['r_peaks'])):
        # Plot R-peaks for each channel
        plot_r_peaks(
            data[:, channel],
            results['r_peaks'][channel]['peaks'],
            f"{filename}_channel_{channel+1}",
            folder
        )
    
    # Save R-peak statistics
    r_peak_stats = pd.DataFrame([
        {
            'Channel': f'Channel_{i+1}',
            'Number_of_Peaks': res['statistics']['n_peaks'],
            'Mean_RR_Interval': res['statistics']['mean_rr'],
            'Std_RR_Interval': res['statistics']['std_rr'],
            'Mean_Heart_Rate': res['statistics']['mean_hr'],
            'Std_Heart_Rate': res['statistics']['std_hr']
        }
        for i, res in enumerate(results['r_peaks'])
    ])
    r_peak_stats.to_csv(f"{folder}/{filename}_r_peak_statistics.csv", index=False)
    
    # Add R-peak information to analysis summary
    with open(f"{folder}/{filename}_clustering_analysis.txt", 'a') as f:
        f.write("\nR-peak Analysis Results\n")
        f.write("-" * 50 + "\n\n")
        
        for i, res in enumerate(results['r_peaks']):
            stats = res['statistics']
            f.write(f"\nChannel {i+1}:\n")
            f.write(f"Number of R-peaks detected: {stats['n_peaks']}\n")
            f.write(f"Mean R-R interval: {stats['mean_rr']:.2f} samples\n")
            f.write(f"Mean heart rate: {stats['mean_hr']:.2f} BPM\n")
            f.write(f"Heart rate variability: {stats['std_hr']:.2f} BPM\n") 