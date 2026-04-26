# CS-7830-Final-Project

## Requirements
 
Install the required dependencies before running:
 
```bash
pip install -r requirements.txt
```
 
## How to Run
 
Run the following command from the root of the `CS-7830-Final-Project` directory:

Windows
```bash
python final_project.py
```

Mac/Unix 
```bash
python3 final_project.py
```
 
This will execute the project models and save all outputs to the `plots/` directory.
 
## Project Directory Structure 
```
├── README.md                                                   # Project overview, setup instructions, and summary of findings
│
├── dataset                                                     # Raw and transformed data storage
│   ├── combine_data_since_2000_PROCESSED_2018-04-26.csv        # Original Kaggle dataset containing historical combine stats
│   └── combine_data_since_2000_PROCESSED_2018-04-26_preprocessed.csv # Cleaned data with encoded targets and scaled features
│
├── final_project.py                                            # Main execution script to run training, evaluation, and plotting
│
├── plots                                                       # Visualizations generated during model testing
│   ├── combine_data_correlation.png                            # Heatmap showing relationships between features
│   ├── mlp_100x100_confusion_matrix.png                        # Confusion Matrix for the [100, 100] architecture
│   ├── mlp_100x100_roc_curve.png                               # AUC-ROC curve for the [100, 100] architecture
│   ├── mlp_128x128x128x128_confusion_matrix.png                # Confusion Matrix for the deepest 4-layer MLP model
│   ├── mlp_128x128x128x128_roc_curve.png                       # Visualizing the True Positive vs False Positive rate for the 128x4 model
│   ├── mlp_128x64x32_confusion_matrix.png                      # Performance breakdown for the funnel-shaped MLP
│   ├── mlp_128x64x32_roc_curve.png                             # ROC curve for the high-performing [128, 64, 32] model
│   ├── mlp_32_confusion_matrix.png                             # Confusion Matrix for the single-layer baseline MLP
│   ├── mlp_32_roc_curve.png                                    # ROC curve for the baseline architecture
│   ├── mlp_64x32_confusion_matrix.png                          # Performance results for the 2-layer MLP
│   ├── mlp_64x32_roc_curve.png                                 # ROC curve for the 2-layer architecture
│   ├── mlp_dif_params[32]_confusion_matrix.png                 # Confusion Matrix for [32] MLP using modified hyperparameters
│   ├── mlp_dif_params[32]_roc_curve.png                        # ROC curve for [32] MLP using modified hyperparameters
│   ├── mlp_dif_params_[100,_100]_confusion_matrix.png          # Confusion Matrix for [100, 100] architecture with modified params
│   ├── mlp_dif_params_[100,_100]_roc_curve.png                 # ROC curve for [100, 100] architecture with modified params
│   ├── mlp_dif_params_[128,_128,_128,_128]_confusion_matrix.png # Confusion Matrix for the deepest MLP using modified params
│   ├── mlp_dif_params_[128,_128,_128,_128]_roc_curve.png       # ROC curve for the deepest MLP using modified params
│   ├── mlp_dif_params_[128,_64,_32]_confusion_matrix.png       # Confusion Matrix for the funnel MLP using modified params
│   ├── mlp_dif_params_[128,_64,_32]_roc_curve.png              # ROC curve for the funnel MLP using modified params
│   ├── mlp_dif_params_[32]_confusion_matrix.png                # Confusion Matrix for the baseline MLP using second set of params
│   ├── mlp_dif_params_[32]_roc_curve.png                       # ROC curve for the baseline MLP using second set of params
│   ├── mlp_dif_params_[64,_32]_confusion_matrix.png            # Confusion Matrix for the 2-layer MLP using modified params
│   ├── mlp_dif_params_[64,_32]_roc_curve.png                   # ROC curve for the 2-layer MLP using modified params
│   ├── multi-layer_perceptron_confusion_matrix.png             # Summary confusion matrix for the primary MLP implementation
│   ├── multi-layer_perceptron_roc_curve.png                    # Combined ROC analysis for the MLP models
│   ├── random_forest_confusion_matrix.png                      # Baseline performance for the ensemble tree model
│   └── random_forest_roc_curve.png                             # ROC curve for the Random Forest baseline
│
├── requirements.txt                                            # Python dependencies (torch, pandas, scikit-learn, etc.)
│
└── src                                                         # Modular source code for the project
    ├── __init__.py                                             # Makes src a Python package for easy imports
    ├── eda.py                                                  # Scripts for Exploratory Data Analysis and statistical profiling
    ├── models.py                                               # PyTorch MLP class definitions and Random Forest configurations
    └── preprocessing.py                                        # Logic for preprocessing the dataset
```