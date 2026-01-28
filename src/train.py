# the main script that glues everything together for the Banking use case.
# This will:
# 1. Generate sample data
# 2.Apply feature engineering
# 3. Train the Random Forest classifier
# 4. Evaluate metrics
# 5. Show a sample forecast (predicted subscription for a few customers)

from data_generation import DataGenerator
from feature_engineering import FeatureEngineer
from model import BankModel
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # -----------------------------
    # 1. Generate sample data
    # -----------------------------
    generator = DataGenerator(n_customers=1000)
    df = generator.generate()
    print("Sample raw data:")
    print(df.head(), "\n")

    # -----------------------------
    # 2. Feature engineering
    # -----------------------------
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    print(f"Feature sample (first 5 rows):\n{X.head()}\n")

    # -----------------------------
    # 3. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -----------------------------
    # 4. Train model
    # -----------------------------
    model = BankModel()
    model.train(X_train, y_train)

    # -----------------------------
    # 5. Evaluate model
    # -----------------------------
    metrics = model.evaluate(X_test, y_test)

    # -----------------------------
    # 6. Sample forecast
    # -----------------------------
    sample_preds = model.predict(X_test.head(5))
    print("\nSample predictions for next 5 customers:")
    for i, val in enumerate(sample_preds, 1):
        status = "Subscribed" if val == 1 else "Not Subscribed"
        print(f"Customer {i}: {status}")

if __name__ == "__main__":
    main()
