# server/Dockerfile
FROM python:3.9

WORKDIR /app

# Install Flower and scikit-learn
RUN pip install flwr scikit-learn

# Copy server code
COPY server.py .

CMD ["python", "server.py"]
