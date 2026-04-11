import pandas as pd
import numpy as np

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

def run_preprocessing() -> pd.DataFrame:
    dataset_path = "dataset/combine_data_since_2000_PROCESSED_2018-04-26.csv"
    initial_combine_df = load_data(dataset_path)
    combine_df = preprocess_data(initial_combine_df)
    combine_df.to_csv("dataset/combine_data_since_2000_PROCESSED_2018-04-26_preprocessed.csv", index=False)
    print(combine_df.head())
    print("\n")

    return combine_df
