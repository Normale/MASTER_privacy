# server/server.py
import flwr as fl

def main():
    # Start Flower server for five rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080", 
    )

if __name__ == "__main__":
    main()
