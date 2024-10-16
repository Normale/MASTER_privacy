# generate_data_model.py
import numpy as np
import pickle
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def generate_data():
    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=1000,  # 1000 samples
        n_features=20,   # 20 features
        n_classes=2,     # Binary classification
        n_informative=15,
        random_state=42
    )
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the test set data (which will be used for evaluation in your federated learning script)
    with open('data.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)
    
    return X_train, y_train

def train_model(X, y):
    # Train a Logistic Regression model
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    
    # Save the model (coefficients and intercept)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

def main():
    # Generate data and train model
    X_train, y_train = generate_data()
    train_model(X_train, y_train)
    
    print("Data and model have been saved to 'data.pkl' and 'model.pkl'")

if __name__ == "__main__":
    main()
