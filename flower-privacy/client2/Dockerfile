# clients/Dockerfile
FROM python:3.9

WORKDIR /app

# Install Flower and scikit-learn
RUN pip install flwr scikit-learn

# Copy client code
COPY client.py .

CMD ["python", "client.py"]
