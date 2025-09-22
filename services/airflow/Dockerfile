FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends docker.io docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -f docker && usermod -aG docker airflow

USER airflow

RUN pip install --no-cache-dir pandas scikit-learn mlflow joblib numpy
