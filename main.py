from pathlib import Path
import matplotlib.pyplot as plt
from step1_analysis import load_cardiogram_data, perform_initial_analysis
from step2_anova import perform_one_way_anova, save_anova_results
from step3_two_factor import perform_two_factor_analysis, save_two_factor_results
from step4_correlation import perform_correlation_analysis, save_correlation_results
from step5_factor import perform_factor_analysis, save_factor_analysis_results
from step6_clustering import perform_clustering_analysis, save_clustering_results
from step7_fourier import perform_fourier_analysis, save_fourier_results

def create_folders():
    """Create folders for different analysis steps"""
    folders = ['step1_analysis', 'step2_anova', 'step3_two_factor', 'step4_correlation', 'step5_factor', 'step6_clustering', 'step7_fourier']
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)
    return folders

def analyze_file(file_path="A6.txt"):
    """Analyze single cardiogram file with detailed output."""
    try:
        # Create output folders
        folders = create_folders()
        filename = Path(file_path).stem
        
        # Step 1: Load data and perform initial analysis
        print("\nStep 1: Loading data and performing initial analysis...")
        data = load_cardiogram_data(file_path)
        initial_results = perform_initial_analysis(data, filename, folders[0])
        
        # Step 2: ANOVA analysis
        print("\nStep 2: Performing ANOVA analysis...")
        anova_results = perform_one_way_anova(data)
        save_anova_results(anova_results, filename, folders[1])
        
        # Step 3: Two-factor analysis
        print("\nStep 3: Performing two-factor analysis...")
        two_factor_results = perform_two_factor_analysis(data)
        save_two_factor_results(two_factor_results, filename, folders[2])
        
        # Step 4: Correlation analysis
        print("\nStep 4: Performing correlation analysis...")
        correlation_results = perform_correlation_analysis(data)
        save_correlation_results(correlation_results, filename, folders[3])
        
        # Step 5: Factor analysis
        print("\nStep 5: Performing factor analysis...")
        factor_results = perform_factor_analysis(data)
        save_factor_analysis_results(factor_results, filename, folders[4])
        
        # Step 6: Clustering analysis
        print("\nStep 6: Performing clustering analysis...")
        clustering_results = perform_clustering_analysis(data)
        save_clustering_results(clustering_results, filename, folders[5], data)
        
        # Step 7: Fourier analysis
        print("\nStep 7: Performing Fourier analysis...")
        fourier_results = perform_fourier_analysis(data)
        save_fourier_results(fourier_results, filename, folders[6])
        
        print("\nAnalysis completed successfully!")
        return {
            'raw_data': data,
            'initial_analysis': initial_results,
            'anova_results': anova_results,
            'two_factor_results': two_factor_results,
            'correlation_results': correlation_results,
            'factor_results': factor_results,
            'clustering_results': clustering_results,
            'fourier_results': fourier_results
        }
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    results = analyze_file("A6.txt")
    plt.show() 