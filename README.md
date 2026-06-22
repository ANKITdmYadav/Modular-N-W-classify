# Network Protocol Classification System

## Overview

The Network Protocol Classification System is an end-to-end Machine Learning and MLOps project designed to automatically classify network protocols using flow-level network traffic features.

The system predicts protocols such as:

* HTTP
* SSL
* DNS
* SSH
* FTP
* ICMP

Instead of inspecting packet payloads, the model learns traffic behavior from network flow statistics, making it useful for network monitoring, traffic analysis, and cybersecurity applications.

---

## Problem Statement

Modern networks generate massive amounts of traffic, and manually identifying protocol types is inefficient and difficult, especially when traffic is encrypted.

The objective of this project is to build an automated machine learning system capable of classifying network protocols using flow-level features such as:

* Flow Duration
* Packet Counts
* Packet Lengths
* Average Packet Size
* TCP Flags
* Protocol Information

---

## Dataset Generation

A hybrid dataset generation strategy was used.

### 1. Cisco Packet Tracer

Custom network traffic was generated for protocols such as:

* DNS
* ICMP
* FTP
* SSH

This allowed controlled protocol simulations and helped understand protocol-specific behavior.

### 2. CICIDS Flow Dataset

Publicly available CICIDS flow-level network traffic data was used to increase dataset diversity and improve generalization.

### Dataset Processing

The generated and collected data underwent:

* Feature Selection
* Feature Alignment
* Data Cleaning
* Missing Value Handling
* Class Balancing
* Label Encoding

The final dataset was used for training protocol classification models.

---

## Project Architecture

```text
Raw Data
    │
    ▼
Data Ingestion
    │
    ▼
Data Validation
    │
    ▼
Data Transformation
    │
    ▼
Model Training
    │
    ▼
MLflow + DagsHub Tracking
    │
    ▼
Model Registry
    │
    ▼
FastAPI Prediction Service
    │
    ▼
Docker Container
    │
    ▼
AWS Deployment
```

---

## Project Structure

```text
networkClassify/

├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   └── model_trainer.py
│
├── entity/
│   ├── config_entity.py
│   └── artifact_entity.py
│
├── pipeline/
│   └── training_pipeline.py
│
├── cloud/
│   └── s3_syncer.py
│
├── utils/
│
├── logging/
│
├── exception/
│
├── templates/
│   └── table.html
│
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
└── setup.py
```

---

## Machine Learning Pipeline

### Data Ingestion

Responsible for collecting data from source files and creating ingestion artifacts.

### Data Validation

Validates:

* Feature consistency
* Schema compliance
* Dataset integrity

Detects potential data drift issues before training.

### Data Transformation

Performs:

* Missing value handling using KNNImputer
* Label encoding
* Feature preparation

Stores transformed data as:

```python
train.npy
test.npy
```

Using NumPy binary format improves loading speed and reduces preprocessing overhead during training.

### Model Training

The training module:

* Loads transformed NumPy arrays
* Trains classification models
* Evaluates performance
* Saves trained models

---

## Hyperparameter Optimization

Optuna was used to automate hyperparameter tuning.

Benefits:

* Improved model performance
* Reduced manual experimentation
* Efficient parameter search

---

## Experiment Tracking

### MLflow

Tracks:

* Parameters
* Metrics
* Models
* Artifacts

Launch UI:

```bash
mlflow ui
```

### DagsHub

Integrated with MLflow for:

* Remote experiment tracking
* Model versioning
* Reproducibility

```python
import dagshub

dagshub.init(
    repo_owner="your_username",
    repo_name="your_repository"
)
```

---

## FastAPI Service

The project exposes prediction endpoints through FastAPI.

Run locally:

```bash
uvicorn app:app --reload
```

or

```bash
python app.py
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

---

## Docker Containerization

Build Image:

```bash
docker build -t network-classify .
```

Run Container:

```bash
docker run -p 8080:8080 network-classify
```

---

## AWS Deployment

### AWS Services Used

* AWS EC2
* AWS ECR
* AWS S3

### S3

Used for storing:

* Model Artifacts
* Training Outputs
* Versioned Models

### ECR

Used as Docker Image Registry.

### EC2

Hosts the deployed FastAPI application.

---

## CI/CD Workflow

GitHub Actions was used to automate deployment.

Pipeline Flow:

```text
GitHub Push
      │
      ▼
GitHub Actions
      │
      ▼
Docker Build
      │
      ▼
Push Image to ECR
      │
      ▼
Pull Image on EC2
      │
      ▼
Deploy Updated Container
```

---

## Features

* End-to-End ML Pipeline
* Modular Architecture
* Custom Logging
* Custom Exception Handling
* Data Validation Layer
* Optuna Hyperparameter Optimization
* MLflow Tracking
* DagsHub Integration
* FastAPI Inference Service
* Docker Containerization
* AWS Deployment
* CI/CD Automation

---

## Tech Stack

### Machine Learning

* Python
* Scikit-Learn
* Optuna

### MLOps

* MLflow
* DagsHub
* Docker
* GitHub Actions

### Backend

* FastAPI

### Cloud

* AWS EC2
* AWS ECR
* AWS S3

### Dataset Generation

* Cisco Packet Tracer
* CICIDS Dataset

---

## Future Improvements

* Real-time packet capture integration
* Streaming predictions using Kafka
* Multi-class deep learning models
* Kubernetes deployment
* Monitoring with Prometheus and Grafana

---

## Key Learning Outcomes

* Building production-grade ML pipelines
* MLOps best practices
* Experiment tracking and model versioning
* AWS deployment workflows
* Network traffic analysis and protocol classification
* Modular software engineering for machine learning systems



![Project Architecture Pipeline](image.png)


![Classification Metric](image-1.png)