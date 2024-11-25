import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_fourier_coefficients(signal, N):
    """
    Compute Fourier coefficients A_j and B_j.
    
    Parameters:
    signal : array-like
        Input signal
    N : int
        Number of points
        
    Returns:
    dict containing A and B coefficients
    """
    # Initialize coefficients
    A = np.zeros(N//2 + 1)
    B = np.zeros(N//2 + 1)
    
    # Time indices
    i = np.arange(N)
    
    # Compute A_0
    A[0] = 2/N * np.sum(signal * np.cos(2*np.pi*0*i/N))
    
    # Compute A_N/2
    A[-1] = 2/N * np.sum(signal * np.cos(np.pi*i))
    
    # Compute other coefficients
    for j in range(1, N//2):
        A[j] = 2/N * np.sum(signal * np.cos(2*np.pi*j*i/N))
        B[j] = 2/N * np.sum(signal * np.sin(2*np.pi*j*i/N))
    
    return {'A': A, 'B': B}

def compute_magnitude_spectrum(A, B):
    """
    Compute magnitude spectrum: c_j = sqrt(A_j^2 + B_j^2)
    """
    return np.sqrt(A**2 + B**2)

def inverse_fourier_transform(A, B, N):
    """
    Perform inverse Fourier transform to reconstruct the signal
    """
    reconstructed = np.zeros(N)
    i = np.arange(N)
    
    # Add DC component (j=0)
    reconstructed += A[0] * np.cos(2*np.pi*0*i/N)
    
    # Add other components
    for j in range(1, len(A)):
        reconstructed += A[j] * np.cos(2*np.pi*j*i/N)
        reconstructed += B[j] * np.sin(2*np.pi*j*i/N)
    
    return reconstructed

def plot_spectrum(spectrum, sampling_rate, title, n_points=None):
    """Plot magnitude spectrum"""
    if n_points is None:
        n_points = len(spectrum)
    
    frequencies = np.fft.fftfreq(len(spectrum)*2, 1/sampling_rate)[:n_points]
    
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, spectrum[:n_points])
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    return plt.gcf()

def perform_fourier_analysis(data, sampling_rate=500):
    """
    Perform complete Fourier analysis for all channels.
    
    Parameters:
    data : array-like
        Input data with shape (n_samples, n_channels)
    sampling_rate : int
        Sampling rate in Hz
    """
    N = len(data)
    n_channels = data.shape[1]
    results = []
    
    for channel in range(n_channels):
        signal = data[:, channel]
        
        # Compute Fourier coefficients
        coeffs = compute_fourier_coefficients(signal, N)
        
        # Compute magnitude spectrum
        spectrum = compute_magnitude_spectrum(coeffs['A'], coeffs['B'])
        
        # Reconstruct signal
        reconstructed = inverse_fourier_transform(coeffs['A'], coeffs['B'], N)
        
        # Compute reconstruction error
        error = np.mean((signal - reconstructed)**2)
        
        results.append({
            'coefficients': coeffs,
            'spectrum': spectrum,
            'reconstructed': reconstructed,
            'error': error
        })
    
    return results

def save_fourier_results(results, filename, folder):
    """Save Fourier analysis results."""
    n_channels = len(results)
    
    # Save spectra for all channels
    spectra = np.array([result['spectrum'] for result in results]).T
    pd.DataFrame(
        spectra,
        columns=[f'Channel_{i+1}' for i in range(n_channels)]
    ).to_csv(f"{folder}/{filename}_spectra.csv")
    
    # Save reconstruction errors
    errors = [result['error'] for result in results]
    pd.DataFrame({
        'Channel': [f'Channel_{i+1}' for i in range(n_channels)],
        'Reconstruction_Error': errors
    }).to_csv(f"{folder}/{filename}_reconstruction_errors.csv")
    
    # Plot and save full spectrum
    plt.figure(figsize=(15, 10))
    for i in range(n_channels):
        plt.plot(results[i]['spectrum'], label=f'Channel {i+1}')
    plt.title('Magnitude Spectrum - All Channels')
    plt.xlabel('Frequency Component')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/{filename}_full_spectrum.png")
    plt.close()
    
    # Plot and save first 200 points
    plt.figure(figsize=(15, 10))
    for i in range(n_channels):
        plt.plot(results[i]['spectrum'][:200], label=f'Channel {i+1}')
    plt.title('Magnitude Spectrum - First 200 Components')
    plt.xlabel('Frequency Component')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{folder}/{filename}_spectrum_200.png")
    plt.close()
    
    # Save analysis summary
    with open(f"{folder}/{filename}_fourier_analysis.txt", 'w') as f:
        f.write("Fourier Analysis Results\n")
        f.write("-" * 50 + "\n\n")
        
        for i in range(n_channels):
            f.write(f"\nChannel {i+1}:\n")
            f.write(f"Reconstruction Error: {results[i]['error']:.6f}\n")
            f.write(f"Max Spectrum Magnitude: {np.max(results[i]['spectrum']):.6f}\n")
            f.write(f"Dominant Frequency Component: {np.argmax(results[i]['spectrum'])}\n") 