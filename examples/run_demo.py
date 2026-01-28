# Calls the full pipeline in train.py

from src.train import main

if __name__ == "__main__":
    """
    Demo script to run the full banking term deposit prediction pipeline.
    - Generates sample customer data
    - Creates features
    - Trains Random Forest model
    - Evaluates model
    - Prints sample predictions
    """
    print("=== Banking Term Deposit Prediction Demo ===\n")
    main()
