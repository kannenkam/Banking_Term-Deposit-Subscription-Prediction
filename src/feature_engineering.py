# This will prepare the data for ML, including encoding categorical variables and scaling numeric variables.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeatureEngineer:
    """
    Prepares features for banking term deposit prediction.
    """

    def __init__(self):
        self.cat_features = ["job", "marital", "loan", "contact", "month"]
        self.num_features = ["age", "balance", "duration"]
        self.encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.scaler = StandardScaler()
        self.fitted = False

    def fit_transform(self, df):
        """
        Fit encoders and scalers on training data and transform features.
        Returns:
            X: transformed feature DataFrame
            y: target series
        """
        df = df.copy()
        y = df["y"]
        X_num = self.scaler.fit_transform(df[self.num_features])
        X_cat = self.encoder.fit_transform(df[self.cat_features])

        X = np.hstack([X_num, X_cat])
        self.fitted = True
        return pd.DataFrame(X), y

    def transform(self, df):
        """
        Transform new data using previously fitted encoder and scaler.
        """
        if not self.fitted:
            raise Exception("FeatureEngineer not fitted yet. Call fit_transform first.")

        df = df.copy()
        X_num = self.scaler.transform(df[self.num_features])
        X_cat = self.encoder.transform(df[self.cat_features])
        X = np.hstack([X_num, X_cat])
        return pd.DataFrame(X)


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    from data_generation import DataGenerator

    # Generate sample data
    generator = DataGenerator(n_customers=500)
    df = generator.generate()

    # Feature engineering
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    print("
