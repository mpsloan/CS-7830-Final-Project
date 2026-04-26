import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_feature_correlation(df: pd.DataFrame, plot_path: str) -> None:
    """
    Visualizes the correlation matrix of the numeric combine features.

    Args:
        df (pd.DataFrame): The preprocessed input DataFrame.
        plot_path (str): The file path to save the plot.
    """
    # Only plot the core numeric combine features
    combine_features = ['Ht', 'Wt', 'Forty', 'Vertical', 'BenchReps', 'BroadJump', 'Cone', 'Shuttle', 'Draft_Position']

    # This computes the Pearson correlation (r) for all numeric columns
    corr_matrix = df[combine_features].corr()

    # Create the Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                center=0,
                linewidths=0.5)

    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def visualize_class_distribution(df: pd.DataFrame, target_col: str, plot_path: str) -> None:
    """
    Visualizes the class distribution of the target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_col (str): The name of the target column.
        plot_path (str): The file path to save the plot.
    """
    class_labels = ['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round']
    counts = df[target_col].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_labels, y=counts.values)
    plt.title('Class Distribution of Draft Position')
    plt.xlabel('Draft Position')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def run_eda(df: pd.DataFrame) -> None:
    """
    Run the various EDA functions defined in this file that our project leverages

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    visualize_feature_correlation(df, "plots/combine_data_correlation.png")
    visualize_class_distribution(df, 'Draft_Position', "plots/combine_data_distribution.png")
