from src.product_dataset import product_dataset
from src.dataset_cleaning import dataset_cleaning
from src.dataset_preprocessing import dataset_preprocessing

import time


def main():

    # Produce a raw dataset for english resume
    product_dataset()

    time.sleep(5)

    # Data cleaned and prepared for pre-processing phase.
    dataset_cleaning()

    time.sleep(5)


    # Data cleaned and prepared for pre-processing phase: 1) NaN Handling 2) Categorical to Numerical 3) Scaling with 2 approches
    dataset_preprocessing()

    


main()