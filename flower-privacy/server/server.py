# server/server.py
from typing import List, Tuple, Dict
import flwr as fl
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from flwr.common import ndarrays_to_parameters, Metrics, Scalar
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from sklearn.metrics import log_loss, accuracy_score


# Example binary classification test dataset (20 features, 200 samples)
X_test = np.random.randn(200, 20)  # 200 samples, 20 features
y_test = np.random.randint(0, 2, 200)  # 200 binary labels (0 or 1)

def evaluate_fn(server_round: int, parameters: List[np.ndarray], config: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Evaluate the global model on a binary classification test set.
    
    # ((int, NDArrays, dict[str, Scalar]) -> (tuple[float, dict[str, Scalar]] | None)) | None = None,
    is basically
     server round (int), parameters (list of numpy arrays), config (dictionary of strings and floats)
        -> returns a tuple of float and dictionary of strings and floats or None
    """
    
    # Rebuild the Logistic Regression model
    model = LogisticRegression(solver='liblinear')
    
    # Set model parameters: 
    # Assume parameters[0] is the coefficient (weights) and parameters[1] is the intercept
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]
    model.classes_ = np.array([0, 1])

    # Make predictions on the test set
    
    accuracy = model.score(X_test, y_test)
    loss = log_loss(y_test, model.predict_proba(X_test))
    return loss, {"accuracy": accuracy}


def load_initial_parameters(model_path):
    # Load the model from a file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(model)
    
    # Get model parameters (weights and intercept)
    # Do not flatten the parameters, keep them in the correct shape
    parameters = [model.coef_, model.intercept_]
    
    # Convert to numpy arrays and wrap in Parameters object
    ndarray_params = [np.array(param) for param in parameters]
    
    return ndarrays_to_parameters(ndarray_params)


def main():
    print("Starting Flower server")
    config = ServerConfig(num_rounds=3)
    
    # Load initial parameters
    initial_parameters = load_initial_parameters('model.pkl')
    
    print("Loaded initial parameters")
    print(initial_parameters)
    
    # evaluate function takes those parameters
    """        return get_model_params(self.model), len(self.X_train), {
            "accuracy": accuracy,
            "loss": train_loss,
            "num_examples": len(self.X_train)  
        }
        """

    # Define strategy
    strategy = FedAvg(
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        min_fit_clients=2,
        # ((int, NDArrays, dict[str, Scalar]) -> (tuple[float, dict[str, Scalar]] | None)) | None = None,
        evaluate_fn=evaluate_fn
    )
    
    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy
    )

if __name__ == "__main__":
    main()