import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# Generate synthetic data for binary classification
X, y = make_classification(n_samples=10000, n_features=20, n_classes=2, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(solver='liblinear', max_iter=1, warm_start=True)

# Number of iterations
n_iterations = 10

for iteration in range(n_iterations):
    # Fit the model (only one iteration at a time)
    model.fit(X_train, y_train)
    
    # Calculate training log loss
    y_train_pred_proba = model.predict_proba(X_train)
    train_loss = log_loss(y_train, y_train_pred_proba)
    
    # Print the log loss for the current iteration
    print(f"Iteration {iteration + 1}/{n_iterations}, Training Log Loss: {train_loss:.4f}")

# Evaluate the final model on the test set
y_test_pred_proba = model.predict_proba(X_test)
test_loss = log_loss(y_test, y_test_pred_proba)
test_accuracy = model.score(X_test, y_test)

print(f"Final Test Log Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
