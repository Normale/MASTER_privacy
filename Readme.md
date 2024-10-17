This is a toy example made from random data.
This example converges relatively quickly so the losses are the same all the time.
You can check the losses being the same if done with just sklearn's Logistic Regression.


To run just do
```sh
docker-compose up
```
in flower-privacy folder.


What happens here?

Server: The server script (server.py) initializes the federated learning server with a specified strategy (FedAvg) and evaluation function. It loads the initial model parameters from model.pkl and starts the server.

Clients: Each client script (client.py, client.py) initializes a Flower client and connects to the server. The clients use the SklearnClient class to handle local training and evaluation.

Data Generation: The data generation script (generate_data.py) creates synthetic binary classification data, splits it into training and test sets, and trains an initial logistic regression model. The model and test data are saved to model.pkl and data.csv, respectively.
