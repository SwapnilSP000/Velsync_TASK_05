"""
Task 05 - ML Model Training Script
Trains a simple classifier using the Iris dataset
and saves it as model.pkl for API consumption.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train simple model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Save trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("🎉 Model trained successfully and saved as model.pkl")


if __name__ == "__main__":
    train_model()
