# KubePulse - *Predict*, *Prevent*, *Perform* { }

---

<img src="https://github.com/user-attachments/assets/2283402f-f86a-447d-aafc-fcce3e2f28ed" alt="KubePulse logo" style="max-width:20%; height:auto;" />


## Project Description

This project is an **AI-powered prediction service** designed to **simulate, monitor, and predict issues** in Kubernetes clusters using real-time metrics (or synthetic data for testing). It leverages **machine learning models** to forecast potential problems such as **resource exhaustion, pod failures, and network anomalies** — all at scale, and cloud-native!

This app serves as the **brain of your Kubernetes fleet**.

---

## 🌐 Application Workflow Overview

### 1. **Simulated or Real Cluster Metrics**
   - Metrics such as CPU utilization, node counts, and temporal features are **generated or received** via API.
   - Simulates 24/7 cluster monitoring.
   - Supports both testing and production pipelines.

### 2. **Machine Learning Inference**
   - A **binary classifier** checks if there's an issue.
   - A **multiclass classifier** determines the issue type.
   - Generates **explanations and recommendations** for remediation.

### 3. **RESTful API**
   - `/predict` – Predict issues based on posted metrics or generate if none are sent.
   - `/simulate/<int:count>` – Simulate multiple clusters and return issues with recommendations.

---

## Pipeline

![DevTrails](https://github.com/user-attachments/assets/79ba5ccf-f850-4a1c-b784-fd9944e853ac)


---

##  Tech Stack

- **Azure**: Source for VM traces (production environment)
- **Pandas / NumPy**: Used for EDA (Exploratory Data Analysis) and feature engineering
- **scikit-learn**: Random Forest classifier model
- **joblib**: For saving and loading the trained model
- **Flask**: API server for Prediction Service and Remediation Agent
- **Google Kubernetes Engine (GKE)**: Model deployment and testing (staging)
- **Azure Kubernetes Service (AKS)**: Production cluster
- **Docker**: Containerizing applications and models
- **Prometheus**: Monitoring
- **Minikube**: Local Kubernetes testing
- **Google Cloud Platform (GCP)**: Hosting environment for microservices
- **Google's Online Boutique (microservice demo app)**: Testing failure scenarios
- **Chaos Mesh**: For injecting failures into microservices


---

## ☁️ Google Kubernetes Engine (GKE) Deployment Workflow

### ⚙️ Step 1: Containerize the Flask App

**Dockerfile**
```Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

Build and push to **Google Container Registry** (GCR):
```bash
docker build -t gcr.io/<PROJECT-ID>/predictor-service .
docker push gcr.io/<PROJECT-ID>/predictor-service
```

---

### 🚢 Step 2: Create Kubernetes Deployment and Service

**`deployment.yaml`**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: service-registery
spec:
  selector:
    matchLabels:
      app: service-registery
  template:  # Move this outside the selector
    metadata:
      labels:
        app: service-registery
    spec:
      containers:
      - name: service-registery
        image: dailycodebuffer/serviceregistery:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8761
```

**`service.yaml`**
```yaml
apiVersion: v1  # Correct API version
kind: Service
metadata:
  name: service-registry-svc
spec:
  selector:
    app: service-registery  # Match the label used in Deployment
  ports:
    - port: 80
      targetPort: 8761
```

Deploy to GKE:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```
---

---

## ⚙️ Microservices Overview

Here's a breakdown of the core microservices involved:

| Service Name         | Functionality |
|----------------------|---------------|
| `frontend`           | UI for the Online Boutique app |
| `load-generator`     | Simulates traffic to the app |
| `failure-injector`   | Triggers chaos experiments (CPU stress, pod kill, etc.) |
| `metrics-collector`  | Scrapes metrics from Prometheus |
| `model-trainer`      | Trains Random Forest model from collected data |
| `failure-predictor`  | Uses trained model to predict failures |
| `monitoring`         | Stack with Prometheus + Grafana |
| `chaos-operator`     | Chaos Mesh operator for fault injection |

---

## 🧱 Application Layer: Online Boutique (Google Microservices Demo)

Deployed using Kubernetes manifests:

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/microservices-demo/main/release/kubernetes-manifests.yaml
```

Key services:
- `frontend`, `cartservice`, `checkoutservice`, `productcatalogservice`, etc.
- These simulate a realistic e-commerce system with multiple dependencies.

---

## 🔁 Load Generator

Continuously generates traffic to simulate normal operations.

### Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: load-generator
spec:
  replicas: 1
  selector:
    matchLabels:
      app: load-generator
  template:
    metadata:
      labels:
        app: load-generator
    spec:
      containers:
      - name: load-generator
        image: alpine/bombardier
        args: ["-c", "20", "-d", "10m", "http://frontend.default.svc.cluster.local"]
```

---

## 💣 Chaos Injection Service

Uses **Chaos Mesh** to simulate failures.

- **Pod Kill**, **CPU Stress**, **Network Delay**
- Scheduled using Cron in the manifests

### Example (Pod Kill):

```yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: pod-kill-frontend
spec:
  action: pod-failure
  mode: one
  selector:
    namespaces:
      - default
    labelSelectors:
      app: frontend
  duration: "60s"
  scheduler:
    cron: "@every 5m"
```

---

## 📊 Metrics Collector Service

A Python service that scrapes data from Prometheus and Chaos Mesh APIs and stores it as JSON or CSV.

### Example:

Stores:
- CPU, memory, restart counts
- Chaos experiment labels (stress, network, pod kill, etc.)
- Timestamps

---



# Random Forest in Kubernetes Failure Prediction: Comparative Analysis and Project Application

## Comparative Analysis of Random Forest vs. Other Machine Learning Algorithms

When selecting the optimal algorithm for our Kubernetes failure prediction system, it's essential to understand how Random Forest compares to alternatives:

| Feature | Random Forest | Decision Trees | Support Vector Machines | Neural Networks | K-Nearest Neighbors |
|---------|---------------|----------------|-------------------------|-----------------|---------------------|
| **Algorithm Type** | Ensemble (multiple trees) | Single tree | Kernel-based | Deep learning | Instance-based |
| **Training Speed** | Moderate | Fast | Slow for large datasets | Very slow | Fast |
| **Prediction Speed** | Moderate | Fast | Fast | Fast after training | Slow for large datasets |
| **Handling Large Datasets** | Excellent | Limited | Poor | Good with sufficient resources | Poor |
| **Handling High-Dimensional Data** | Excellent | Good | Good | Excellent | Poor |
| **Risk of Overfitting** | Low | High | Moderate | High without regularization | High |
| **Interpretability** | Moderate (feature importance) | High | Low | Very low | Moderate |
| **Hyperparameter Tuning Needed** | Minimal | Minimal | Extensive | Extensive | Minimal |
| **Handles Missing Values** | Yes | Yes | No | No | No |
| **Handles Non-Linear Relationships** | Yes | Yes | Yes (with kernels) | Yes | Yes |
| **Computational Resources** | Moderate | Low | High | Very high | Low |

## Advantages of Random Forest for Our Kubernetes Prediction Project

After analyzing the comparative strengths of various algorithms, Random Forest emerges as an ideal choice for our Kubernetes failure prediction system for several key reasons:

### 1. High Accuracy Through Ensemble Learning
Random Forest aggregates predictions from multiple decision trees, each trained on different subsets of our Kubernetes metrics data. This ensemble approach significantly improves prediction accuracy compared to single-model approaches, which is crucial for reliable failure forecasting.

### 2. Robustness to Noise and Outliers
Kubernetes environments generate noisy metrics with frequent spikes and anomalies. Random Forest's inherent resistance to noise makes it particularly well-suited for handling the variability in cluster telemetry data without compromising prediction quality.

### 3. Minimal Hyperparameter Tuning Required
During our 45-day hackathon timeline, Random Forest offers a significant advantage by providing strong "out-of-the-box" performance with default settings. This allows us to focus more on data collection and feature engineering rather than extensive model tuning.

### 4. Feature Importance Assessment
Random Forest automatically calculates feature importance, helping us identify which Kubernetes metrics (CPU utilization, memory usage, pod restart counts, etc.) are most predictive of upcoming failures. This capability provides actionable insights for both model refinement and potential remediation strategies.

### 5. Handles Missing Values Effectively
In production Kubernetes environments, metrics collection can sometimes be incomplete due to node issues or monitoring disruptions. Random Forest can naturally handle these missing values without requiring complex imputation strategies.

### 6. Balanced Bias-Variance Tradeoff
By combining multiple trees trained on different data subsets, Random Forest effectively manages the bias-variance tradeoff, producing models that generalize well to unseen Kubernetes failure patterns.

### 7. Non-Parametric Nature
Random Forest makes no assumptions about the underlying distribution of Kubernetes metrics or the relationships between variables. This flexibility allows it to capture complex, non-linear patterns in cluster behavior that might indicate impending failures.

### 8. Effective with Imbalanced Data
Kubernetes failures are relatively rare events compared to normal operation, creating an imbalanced dataset. Random Forest handles this class imbalance effectively through its bootstrap sampling approach, ensuring minority classes (failure events) are adequately represented in the model.

## Implementation in Our Project

For our Kubernetes failure prediction system, we're implementing Random Forest in the following ways:

1. **Baseline Model**: We're using Random Forest as our primary baseline model, targeting 5-minute prediction windows for various failure types.

2. **Feature Selection**: Leveraging Random Forest's feature importance metrics to identify the most predictive Kubernetes metrics, allowing us to streamline data collection.

3. **Time-Series Windows**: Processing Kubernetes metrics into 3-minute sliding windows (with 15-second intervals) as input features, with Random Forest predicting whether a failure will occur in the subsequent 5 minutes.

4. **Comparative Benchmark**: Using Random Forest performance (expected ~73-78% accuracy based on research) as a benchmark against which to compare more complex models like Temporal Convolutional Networks.

5. **Interpretability Layer**: Utilizing feature importance scores to provide human-readable explanations of why the system predicts specific failures, enhancing trust and adoption.

---

### 🔄 Step 3: Kubernetes Monitoring & Autoscaling (Optional but recommended)

**Enable Horizontal Pod Autoscaler**
```bash
kubectl autoscale deployment predictor-service --cpu-percent=50 --min=2 --max=10
```

**Enable GKE Monitoring with Stackdriver**
- Use `kubectl get pods`, `kubectl logs`, and Google Cloud Console to monitor deployments.

---

### 🔄 Step 4: CI/CD 
- Used **Cloud Build** or **GitHub Actions** for continuous integration.
- Used **Kustomize** or **Helm** for deployment templating.

---


## 🔁 Dev & Test Workflow: Azure + GKE

| Environment | Cloud | Use Case                       |
|-------------|-------|-------------------------------|
| **Azure**   | AKS   | Production deployment          |
| **GCP**     | GKE   | Staging, simulated stress test |

> Push once, deploy to both with `kubectl apply`.

You can also use **multi-cloud GitHub Actions** to build, test, and deploy to both environments in parallel!

---


## 🧪 Testing Locally

```bash
python app.py
curl http://localhost:5000/predict
curl http://localhost:5000/simulate/5
```

---

## 🧠 Example Prediction Response

```json
{
  "timestamp": "2025-04-16T10:10:00Z",
  "cluster_id": "5",
  "prediction": {
    "will_have_issue": true,
    "issue_probability": 0.87,
    "issue_type": "pod_failure",
    "recommendation": {
      "description": "Pods are failing or being terminated unexpectedly",
      "actions": [
        "Investigate pod logs for error messages",
        "Check for memory leaks or OOM kills",
        "Restart affected deployments",
        "Implement liveness and readiness probes"
      ]
    }
  },
  "metrics": {
    "cpu_utilization": 79.32,
    "node_count": 3
  }
}
```

---

## 💡 Real-World Use Cases

- **Preemptive Autoscaling**: Act before disaster strikes!
- **Incident Response**: Integrate with alerting tools.
- **Cost Optimization**: Avoid over-provisioning.
- **DevSecOps**: Detect anomalies and apply remediation.

---

## 🏁 Conclusion

With this project, we're not just deploying a Flask app. we're deploying **an intelligent observability engine** that watches over our Kubernetes infrastructure like a hawk. Integrated with GKE, it's scalable, resilient, and production-ready.



## Dataset :

[Azure Public Dataset - VM Traces](https://github.com/Azure/AzurePublicDataset)


- Source: https://github.com/Azure/AzurePublicDataset
- Description: Contains VM workload traces that can be translated to Kubernetes workloads
- Advantage: Well-documented and includes CPU, memory usage patterns



To set up this in youe environment, do as follow :

```

# Create a data directory if it doesn't exist
mkdir -p data

# Download Azure Public Dataset V2 (more recent than V1)
# You can use one of the links from AzurePublicDatasetLinksV2.txt
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/vmtable/vmtable.csv.gz -P data/
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/deployments/deployments.csv.gz -P data/
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz -P data/

# Uncompress files
gunzip data/*.gz


```



## Developers 


- [@Sai Nivedh V](https://github.com/SaiNivedh26)
- [@Baranidharan S](https://github.com/thespectacular314)
- [@Roshan T](https://github.com/Twinn-github09)
- [@Hari Heman](https://github.com/MAD-MAN-HEMAN)
- [@Kavinesh](https://github.com/Kavinesh11)
