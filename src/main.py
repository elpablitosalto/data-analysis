from data_loader import load_data, clean_data, get_data_info
from analysis import calculate_basic_stats, analyze_correlations, analyze_categorical, detect_outliers
from visualization import plot_numeric_distributions, plot_correlation_heatmap, plot_categorical_counts, plot_boxplots, plot_scatter_matrix
import os


def main():
    # Ensure the reports directory exists
    os.makedirs('reports', exist_ok=True)

    # Load and clean data
    file_path = 'data/sample_data.csv'  # Update with your CSV file path
    df = load_data(file_path)
    df_clean = clean_data(df)

    # Get data information
    info = get_data_info(df_clean)
    print("Data Information:", info)

    # Perform analysis
    stats = calculate_basic_stats(df_clean)
    print("Basic Statistics:", stats)

    correlations = analyze_correlations(df_clean)
    print("Correlations:", correlations)

    categorical_analysis = analyze_categorical(df_clean)
    print("Categorical Analysis:", categorical_analysis)

    outliers = detect_outliers(df_clean)
    print("Outliers:", outliers)

    # Generate visualizations
    plot_numeric_distributions(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_categorical_counts(df_clean)
    plot_boxplots(df_clean)
    plot_scatter_matrix(df_clean)


if __name__ == "__main__":
    main() 