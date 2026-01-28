# This will simulate sample customer data so your repo is fully runnable without real data.

import pandas as pd
import numpy as np

class DataGenerator:
    """
    Generates simulated banking customer data for term deposit subscription prediction.
    """

    def __init__(self, n_customers=1000, seed=42):
        """
        Args:
            n_customers (int): Number of customers to simulate
            seed (int): Random seed for reproducibility
        """
        self.n_customers = n_customers
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self):
        """
        Returns a pandas DataFrame with the following columns:
        - age: customer age (18-70)
        - job: customer job type
        - marital: marital status
        - balance: bank account balance
        - loan: whether customer has a personal loan (yes/no)
        - contact: contact type (cellular/telephone)
        - day: last contact day of the month (1-31)
        - month: last contact month (Jan-Dec)
        - duration: last contact d
