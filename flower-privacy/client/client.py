import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class SklearnClient(fl.client.NumPyClient):
    def __init__(self):
        # Initialize a simple Logistic Regression model
        self.model = LogisticRegression(solver='liblinear')

        # Simulated dataset (replace with real data if available)
        self.X, self.y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def get_parameters(self, config):
        # Ensure the model is fitted before accessing coef_ and intercept_
        if not hasattr(self.model, 'coef_'):
            self.model.fit(self.X_train, self.y_train)
        # Return model parameters (weights and intercept)
        return self.model.coef_.flatten().tolist() + self.model.intercept_.tolist()

    def set_parameters(self, parameters):
        # Set model parameters (weights and intercept)
        n_features = self.X.shape[1]
        self.model.coef_ = np.array(parameters[:-1]).reshape(1, n_features)
        self.model.intercept_ = np.array(parameters[-1:])

    def fit(self, parameters, config):
        # Fit model on local data
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Evaluate model on test data
        self.set_parameters(parameters)
        loss = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {}

def main():
    # Start the Flower client
    client = SklearnClient()
    fl.client.start_client(server_address="flower-server:8080", client=client.to_client())
    print("Client closed")

if __name__ == "__main__":
    main()
