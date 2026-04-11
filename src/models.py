import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy import ndarray
import pandas as pd


# Try making the src run on gpu, else run on cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_logistic_regression(
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray
) -> LogisticRegression:
    """
    Trains and evaluates a Logistic Regression classifier.

    Args:
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

    Args:
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

    Args:
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

    Args:
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

    Args:
        df (pd.DataFrame): preprocessed input dataframe
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
        df (pd.DataFrame): The preprocessed input DataFrame containing features and target.
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
        print(f"\n" + "!" * 50)
        print(f" TESTING ARCHITECTURE: {arch}")
        print("!" * 50)

        # Initialize FlexibleMLP (assumes the class is defined in your script)
        model = FlexibleMLP(input_dim, arch, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training Loop
        model.train()
        for epoch in range(150):  # Increased slightly for better convergence
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

def run_models(df: pd.DataFrame):
    """
    Run the various models defined in this file that our project leverages

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    run_simple_models(df, 'Draft_Position')
    run_mlp(df, 'Draft_Position')
