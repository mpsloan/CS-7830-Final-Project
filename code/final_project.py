import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from numpy import ndarray

# Try making the code run on gpu, else run on cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def visualize_feature_correlation(df: pd.DataFrame, labels: list, plot_path: str):
    """
    Visualizes the correlation matrix of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
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

def update_round(round_picked: float) -> int:
    """
    Update the round value based on specific criteria.

    Args:
        round_picked (float): The round the player was picked in.

    Returns:
        int: The updated round value.
    """

    # If there is no round, then they were undrafted, specify as 0
    if np.isnan(round_picked):
        return 0

    # If the player was drafted in the first two rounds it is an early pick, specify as 1
    elif round_picked == 1 or round_picked == 2:
        return 1

    # If the player was drafted in the 3rd or 4th rounds it is an mid-round pick, specify as 2
    elif round_picked == 3 or round_picked == 4:
        return 2

    # If the player was drafted past the 4th round it is an late-round pick, specify as 3
    else:
        return 3

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Args: 
        file_path (str): The path to the CSV file to load.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    
    data = pd.read_csv(file_path)

    return data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and encoding categorical variables.

    Args:
        df (pd.DataFrame): The input data to preprocess.

    Returns:
        pd.DataFrame: The preprocessed data.
    """

    # Drop columns that are not needed
    cols_to_drop = ['Player', 'Team', 'Pick', 'Pfr_ID', 'Year']
    df.drop(columns=cols_to_drop, inplace=True)

    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)

    # Update the round picked to 0, if not drafted, 1 if early, 2 if mid, 3 if late
    df['Round'] = df['Round'].apply(update_round)
    df.rename(columns={'Round': 'Draft_Position'}, inplace=True)

    # We use median because it is robust to outliers in athletic metrics
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    # Fill NaNs with the median of each column
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    return df

def run_logistic_regression(
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray
) -> LogisticRegression:
    """
    Trains and evaluates a Logistic Regression classifier.

    Parameters:
        X_train (ndarray): training features
        X_test (ndarray): test features
        y_train (ndarray): training labels
        y_test (ndarray): test labels

    Returns:
        LogisticRegression: trained LogisticRegression instance
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    target_names = ['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round']
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))
    print("\n")
    return model

def run_knn(
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray,
        n_neighbors: int = 5
) -> KNeighborsClassifier:
    """
    Trains and evaluates a K-Nearest Neighbors classifier.

    Parameters:
        X_train (ndarray): training features
        X_test (ndarray): test features
        y_train (ndarray): training labels
        y_test (ndarray): test labels
        n_neighbors (int): number of neighbors to use. Default is 5.

    Returns:
        KNeighborsClassifier: trained KNeighborsClassifier instance
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    target_names = ['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round']
    print("KNeighbors Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))
    print("\n")
    return model

def run_svm(
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray,
        kernel: str = 'rbf'
) -> SVC:
    """
    Trains and evaluates a Support Vector Machine classifier.

    Parameters:
        X_train (ndarray): training features
        X_test (ndarray): test features
        y_train (ndarray): training labels
        y_test (ndarray): test labels
        kernel (str): kernel type to use ('linear', 'rbf', 'poly'). Default is 'rbf'.

    Returns:
        SVC: trained SVC instance
    """
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    target_names = ['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round']
    print("SVC Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))
    print("\n")
    return model

def run_random_forest(
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray,
        n_estimators: int = 100
) -> RandomForestClassifier:
    """
    Trains and evaluates a Random Forest classifier.

    Parameters:
        X_train (ndarray): training features
        X_test (ndarray): test features
        y_train (ndarray): training labels
        y_test (ndarray): test labels
        n_estimators (int): number of trees in the forest. Default is 100.

    Returns:
        RandomForestClassifier: trained RandomForestClassifier instance
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    target_names = ['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round']
    print("Random Forest Classification Report:")
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))
    print("\n")
    return model

def run_simple_models(df: pd.DataFrame, target_col: str) -> None:
    """
    Splits data once and runs all four classifiers on the same train/test sets.

    Parameters:
        df (pd.DataFrame): input dataframe
        target_col (str): name of the target column

    Returns:
        None
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_knn(X_train, X_test, y_train, y_test, n_neighbors=3)
    run_svm(X_train, X_test, y_train, y_test, kernel='rbf')
    run_random_forest(X_train, X_test, y_train, y_test, n_estimators=50)

class FlexibleMLP(nn.Module):
    """
    A flexible multi-layer perceptron (MLP) class that can adapt its architecture.
    """

    def __init__(self, input_size, hidden_layers, num_classes):
        """
        Initialize the FlexibleMLP model.

        Args:
            input_size (int): The number of input features.
            hidden_layers (list): A list containing the number of units in each hidden layer.
            num_classes (int): The number of output classes.
        """

        super(FlexibleMLP, self).__init__()
        layers = []
        in_dim = input_size
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        return self.network(x)

def run_mlp(df: pd.DataFrame, target_col: str) -> dict:
    """
    Run the test on the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target.
        target_col (str): The name of the target column.

    Returns:
        dict: A dictionary containing the results of the test.
    """

    # Grab features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split data
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X.values, y.values, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale and move to GPU
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train_raw), dtype=torch.float32).to(device)
    X_test = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32).to(device)
    
    y_train = torch.tensor(y_train_raw, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test_raw, dtype=torch.long).to(device)
    
    # --- TEST DIFFERENT ARCHITECTURES ---
    architectures = [
        [32],               
        [64, 32],           
        [128, 64, 32],     
        [100, 100]         
    ]
    
    results = {}
    num_classes = len(np.unique(y_train_raw))
    input_dim = X_train.shape[1]

    for arch in architectures:
        print(f"\n" + "!"*50)
        print(f" TESTING ARCHITECTURE: {arch}")
        print("!"*50)
        
        # Initialize FlexibleMLP (assumes the class is defined in your script)
        model = FlexibleMLP(input_dim, arch, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        model.train()
        for epoch in range(150): # Increased slightly for better convergence
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Evaluation Mode
        model.eval()
        with torch.no_grad():
            # Get predictions on GPU
            test_outputs = model(X_test)
            _, y_pred_tensor = torch.max(test_outputs, dim=1)
            
            # Move back to CPU for metrics
            y_true = y_test.cpu().numpy()
            y_pred = y_pred_tensor.cpu().numpy()

            # Calculate Metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            print(f"\nFinal Training Loss: {loss.item():.4f}")
            print("-" * 30)
            print(f"Test Accuracy: {acc:.4f}")
            print(f"Macro F1-Score: {f1:.4f}")
            print("-" * 30)
            
            target_names = ['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round']
            print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
            
            results[str(arch)] = f1

    return results

def final_project_pipeline():
    """
    Run the final project
    """

    # Seed to make results in report reproducible
    random.seed(16)

    dataset_path = "../dataset/combine_data_since_2000_PROCESSED_2018-04-26.csv"
    combine_df = load_data(dataset_path)
    combine_df = preprocess_data(combine_df)
    combine_df.to_csv("../dataset/combine_data_since_2000_PROCESSED_2018-04-26_preprocessed.csv", index=False)
    visualize_feature_correlation(combine_df, combine_df.columns.tolist(), "../plots/combine_data_correlation.png")
    print(combine_df.head())
    print("\n")
    run_simple_models(combine_df, 'Draft_Position')
    run_mlp(combine_df, 'Draft_Position')


if __name__ == "__main__":
    final_project_pipeline()
