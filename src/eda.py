import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_feature_correlation(df: pd.DataFrame, labels: list, plot_path: str) -> None:
    """
    Visualizes the correlation matrix of the DataFrame.

    Args:
        df (pd.DataFrame): The preprocessed input DataFrame.
        labels (list): The labels for each sample.
        plot_path (str): The file path to save the plot.
    """
    # This computes the Pearson correlation (r) for all numeric columns
    corr_matrix = df[labels].corr()

    # Create the Heatmap
    plt.figure(figsize=(25, 25))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                center=0,
                linewidths=0.5)

    plt.title('Feature Correlation Heatmap')
    plt.savefig(plot_path)
    plt.close()

def run_eda(df: pd.DataFrame) -> None:
    """
    Run the various EDA functions defined in this file that our project leverages

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    visualize_feature_correlation(df, df.columns.tolist(), "plots/combine_data_correlation.png")
