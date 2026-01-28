# This will handle training, prediction, and evaluation of the Random Forest classifier.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BankModel:
    """
    Encapsulates ML model for banking term deposit prediction.
    """

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.is_trained = False

    def train(self, X_train, y_train):
        """
        Train the Random Forest classifier.
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training completed.")

    def predict(self, X):
        """
        Predict using the trained model.
        """
        if not self.is_trained:
            raise Exception("Model not trained yet. Call train() first.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using Accuracy, Precision, Recall, and F1-score.
        Returns:
            metrics (dict): dictionary of evaluation metrics
        """
        preds = self.predict(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-score": f1_score(y_test, preds)
        }
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f}")
        return metrics


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    from data_generation import DataGenerator
    from feature_engineering import FeatureEngineer
    from sklearn.model_selection import train_test_split

    # Generate sample data
    generator = DataGenerator(n_customers=500)
    df = generator.generate()

    # Feature engineering
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate model
    model = BankModel()
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
