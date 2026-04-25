import copy  # Needed to save the best model weights
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix,
                             accuracy_score, f1_score)
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

class MLP(nn.Module):
    """
    A flexible multi-layer perceptron (MLP) class that can adapt its architecture.
    """

    def __init__(self, input_size: int, hidden_layers: list, num_classes: int, dropout_prob: float):
        """
        Initialize the MLP model.

        Args:
            input_size (int): The number of input features.
            hidden_layers (list): A list containing the number of units in each hidden layer.
            num_classes (int): The number of output classes.
            dropout_prob (float): The dropout probability for regularization.
        """

        super(MLP, self).__init__()
        layers = []
        in_dim = input_size

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        [100, 100],
        [128, 128, 128, 128]
    ]

    num_classes = len(np.unique(y_train_raw))
    input_dim = X_train.shape[1]

    for arch in architectures:
        print(f"\n" + "!" * 50)
        print(f" TESTING ARCHITECTURE: {arch}")
        print("!" * 50)

        model = MLP(input_dim, arch, num_classes, dropout_prob=0.2).to(device)

        # Give higher weight to classes with lower support
        weights = torch.tensor([1.0, 2.2, 2.1, 1.7]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

        # Training Loop
        model.train()
        for epoch in range(250): 
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            # scheduler.step()

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

def run_best_mlp(
        X_train_raw: ndarray,
        X_test_raw: ndarray,
        y_train_raw: ndarray,
        y_test_raw: ndarray,
        feature_names: list
) -> dict:
    """
    Runs optimized MLP training and returns the single best performing model and its metadata.

    Args:
        X_train_raw (ndarray): raw training features
        X_test_raw (ndarray): raw test features
        y_train_raw (ndarray): raw training labels
        y_test_raw (ndarray): raw test labels
        feature_names (list):  a list containing the names of the features from the preprocessed dataframe

    Returns:
        Dictionary containing the best performing model and its metadata.
    """

    # Use forward feature selection to see if this improves the model
    print("Running Forward Selection to optimize feature set...")
    selector_model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
    sfs = SequentialFeatureSelector(
        selector_model, 
        n_features_to_select=12, 
        direction='forward', 
        scoring='f1_macro', 
        cv=3, 
        n_jobs=-1
    )
    sfs.fit(X_train_raw, y_train_raw)

    # Apply feature mask to both train and test
    feature_mask = sfs.get_support()
    selected_features = [name for name, selected in zip(feature_names, feature_mask) if selected]
    print(f"Best Features: {selected_features}")

    X_train_selected = X_train_raw[:, feature_mask]
    X_test_selected = X_test_raw[:, feature_mask]

    # Feature Scaling and creating test tensors
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train_selected), dtype=torch.float32).to(device)
    X_test = torch.tensor(scaler.transform(X_test_selected), dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_raw, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test_raw, dtype=torch.long).to(device)

    # Architectures we want to test
    architectures = [[32], [64, 32], [128, 64, 32], [100, 100], [128, 128, 128, 128]]
    num_classes = len(np.unique(y_train_raw))
    input_dim = X_train.shape[1]

    # Give higher weight to classes with lower support
    weights = torch.tensor([1.0, 2.2, 2.1, 1.7]).to(device)

    best_f1 = -1.0
    best_overall_results = {
        "model": None,
        "architecture": None,
        "features": selected_features,
        "f1_score": 0,
        "accuracy": 0,
        "report": ""
    }

    # Iterate over different architectures
    for arch in architectures:
        print(f"\nTesting Architecture: {arch}")

        # Initialize Model, Loss, Optimizer, and Scheduler
        model = MLP(input_dim, arch, num_classes, dropout_prob=0.2).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

        # Model training mode
        model.train()

        # Training loop with epochs
        for epoch in range(250): 
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.detach())

        # Model evaluation mode
        model.eval()

        with torch.no_grad():

            # Get model predictions
            test_outputs = model(X_test)
            _, y_pred_tensor = torch.max(test_outputs, dim=1)
            y_true = y_test.cpu().numpy()
            y_pred = y_pred_tensor.cpu().numpy()

            # Calculate Metrics
            current_f1 = f1_score(y_true, y_pred, average='macro')
            current_acc = accuracy_score(y_true, y_pred)
            current_report = classification_report(y_true, y_pred, target_names=['Undrafted', 'Early Pick', 'Mid-Round', 'Late-Round'], zero_division=0)

            # Check if this is the best model 
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_overall_results.update({
                    "model": copy.deepcopy(model),  # Save the actual model
                    "architecture": arch,
                    "f1_score": current_f1,
                    "accuracy": current_acc,
                    "report": current_report
                })
                print(f"--> New Best Model Found! (F1: {current_f1:.4f})")

    # Output best model results
    print("\n" + "="*49)
    print("WINNING MODEL CONFIGURATION")
    print("="*49)
    print(f"Architecture: {best_overall_results['architecture']}")
    print(f"Best Macro F1: {best_overall_results['f1_score']:.4f}")
    print(f"Accuracy: {best_overall_results['accuracy']:.4f}")
    print("\nDetailed Report:\n", best_overall_results['report'])

    return best_overall_results

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
    run_best_mlp(X_train_raw, X_test_raw, y_train_raw, y_test_raw, features)
