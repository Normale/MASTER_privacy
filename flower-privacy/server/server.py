# server/server.py
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

def load_initial_parameters():
    # Initialize a simple Logistic Regression model
    model = LogisticRegression(solver='liblinear')
    
    # Simulated dataset (replace with real data if available)
    X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    # Return model parameters (weights and intercept)
    return model.coef_.flatten().tolist() + model.intercept_.tolist()

def main():
    config = ServerConfig(num_rounds=3)
    # Load initial parameters
    initial_parameters = load_initial_parameters()
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        initial_parameters=initial_parameters,
        fraction_fit=0.5,
        min_fit_clients=1,
        min_available_clients=1,
        evaluate_metrics_aggregation_fn=lambda metrics: np.mean([m["accuracy"] for m in metrics]),
    )
    
    # Start Flower server for five rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy
    )

if __name__ == "__main__":
    main()