import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from flwr.common import ndarrays_to_parameters, Metrics, Scalar

def get_model_params(model):
    parameters = [model.coef_, model.intercept_]
    
    # Convert to numpy arrays and wrap in Parameters object
    ndarray_params = [np.array(param) for param in parameters]
    
    return ndarray_params

def set_model_params(model, params):
    print("Setting model parameters")
    
    # The first element of params is coef_ and the second is intercept_
    model.coef_ = np.array(params[0])
    if model.fit_intercept:
        model.intercept_ = np.array(params[1])
    
    return model


def set_initial_params(model):
    n_classes = 2  # Number of classes in dataset
    n_features = 20  # Number of features in dataset
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

class SklearnClient(fl.client.NumPyClient):
    def __init__(self):
        print("Initializing SklearnClient")
        # Initialize a simple Logistic Regression model
        self.model = LogisticRegression(solver='liblinear', max_iter=1)
        set_initial_params(self.model)
        self.X, self.y = make_classification(n_samples=10000, n_features=20, n_classes=2, n_informative=15)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)

    def fit(self, parameters, config):
        self.model = set_model_params(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        train_loss = log_loss(self.y_train, self.model.predict_proba(self.X_train))
        return get_model_params(self.model), len(self.X_train), {
            "loss": train_loss,
        }
    
    def evaluate(self, parameters, config):
        self.model = set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def main():
    print("Starting Flower client")
    # Start the Flower client
    client = SklearnClient()
    print("Client started")
    fl.client.start_client(server_address="flower-server:8080", client=client.to_client())
    print("Client closed")

main()
