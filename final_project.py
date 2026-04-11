import random

from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.models import run_models


def final_project_pipeline():
    """
    Run the final project
    """

    # Seed to make results in report reproducible
    random.seed(16)

    combine_df = run_preprocessing()
    run_eda(combine_df)
    run_models(combine_df)


if __name__ == "__main__":
    final_project_pipeline()
