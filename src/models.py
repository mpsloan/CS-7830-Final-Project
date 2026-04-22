import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import label_binarize

# Try making the src run on gpu, else run on cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# make PyTorch results reproducible as well
torch.manual_seed(42)
torch.mps.manual_seed(42) if torch.backends.mps.is_available() else None

def run_random_forest(
        X_train: ndarray,
        X_test: ndarray,
        y_train: ndarray,
        y_test: ndarray,
        features: list,
        n_estimators: int = 100,
) -> None:
    """
    Trains and evaluates a Random Forest classifier.

    Args:
        X_train (ndarray): training features
        X_test (ndarray): test features
        y_train (ndarray): training labels
        y_test (ndarray): test labels
        features (list): A list containing the names of the features from the preprocessed dataframe
        n_estimators (int): number of trees in the forest, default is 100

    """
    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Run model and compute metrics
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    compute_metrics(y_test, predictions, "Random Forest")

    # Print the most important features random forest is leveraging
    importance_pairs = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print("Random Forest Most Important Features: \n")
    for name, score in importance_pairs:
        print(f"{name}: {score:.2f}")

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


def run_mlp(
        X_train_raw: ndarray,
        X_test_raw: ndarray,
        y_train_raw: ndarray,
        y_test_raw: ndarray
) -> None:
    """
    Trains and evaluates a FlexibleMLP across multiple architectures.

    Args:
        X_train_raw (ndarray): raw training features, will be scaled and converted to tensors
        X_test_raw (ndarray): raw test features, will be scaled and converted to tensors
        y_train_raw (ndarray): raw training labels, will be converted to tensors
        y_test_raw (ndarray): raw test labels, will be converted to tensors
    """

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

            print(f"\nFinal Training Loss: {loss.item():.4f}")
            print("-" * 30)

            # Create string to display various architectures cleanly (filenames and console output)
            arch_str = 'x'.join(map(str, arch))
            compute_metrics(y_true, y_pred, f"MLP_{arch_str}")

def compute_metrics(y_test: ndarray, predictions: ndarray, model_name: str) -> dict:
    """
    Computes and displays evaluation metrics for our classification models.

    Args:
        y_test (ndarray): true labels
        predictions (ndarray): predicted labels
        model_name (str): name of the model for display purposes

    Returns:
        dict: dictionary containing macro_f1, weighted_f1, accuracy, and auc_roc
    """
    target_names = ['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round']
    classes = [0, 1, 2, 3]

    # Classification Report
    print(f"\n{model_name} Classification Report:")
    report = classification_report(y_test, predictions, target_names=target_names, zero_division=0, output_dict=True)
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))

    # AUC-ROC (one-vs-rest for multiclass)
    y_test_binarized = label_binarize(y_test, classes=classes)
    predictions_binarized = label_binarize(predictions, classes=classes)
    auc_roc = roc_auc_score(y_test_binarized, predictions_binarized, average='macro', multi_class='ovr')
    print(f"Macro AUC-ROC: {auc_roc:.4f}")

    # ROC Curve Plot (one curve per class)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], predictions_binarized[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} ROC Curve')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_roc_curve.png")
    plt.close()

    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"plots/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'auc_roc': auc_roc
    }


def run_models(df: pd.DataFrame) -> None:
    """
    Splits data once and runs all models on the same train/test sets.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
    """
    # Split the data once and pass the splits into the models
    X = df.drop(columns=['Draft_Position'])
    y = df['Draft_Position']
    features = X.columns.tolist()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X.values, y.values, test_size=0.2, stratify=y, random_state=42
    )

    run_random_forest(X_train_raw, X_test_raw, y_train_raw, y_test_raw, features)
    run_mlp(X_train_raw, X_test_raw, y_train_raw, y_test_raw)
