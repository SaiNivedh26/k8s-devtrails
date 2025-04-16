# k8s-devtrails

Awesome! Here's a **README** file with an **exaggerated, high-level explanation** of your app, integrating it with **Google Kubernetes Engine (GKE)** and detailing the **complete Kubernetes (K8s) workflow**.

---

# 🚀 AI-Driven Kubernetes Cluster Health Predictor

## 🤖 Project Description

This project is an **AI-powered prediction service** designed to **simulate, monitor, and predict issues** in Kubernetes clusters using real-time metrics (or synthetic data for testing). It leverages **machine learning models** to forecast potential problems such as **resource exhaustion, pod failures, and network anomalies** — all at scale, and cloud-native!

This app serves as the **brain of your Kubernetes fleet**, empowering SREs, DevOps engineers, and platform teams to be *proactive* instead of *reactive*.

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

## 🧠 Tech Stack

- **Flask**: API server
- **scikit-learn / joblib**: ML model loading and inference
- **Pandas/Numpy**: Feature engineering
- **Docker**: Containerization
- **Kubernetes (GKE)**: Orchestration and deployment
- **Google Cloud Platform**: Hosting the entire ML-powered microservice

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
  name: predictor-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: predictor
  template:
    metadata:
      labels:
        app: predictor
    spec:
      containers:
      - name: predictor
        image: gcr.io/<PROJECT-ID>/predictor-service
        ports:
        - containerPort: 5000
```

**`service.yaml`**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: predictor-service
spec:
  type: LoadBalancer
  selector:
    app: predictor
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
```

Deploy to GKE:
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

---

### 🔄 Step 3: Kubernetes Monitoring & Autoscaling (Optional but recommended)

**Enable Horizontal Pod Autoscaler**
```bash
kubectl autoscale deployment predictor-service --cpu-percent=50 --min=2 --max=10
```

**Enable GKE Monitoring with Stackdriver**
- Use `kubectl get pods`, `kubectl logs`, and Google Cloud Console to monitor deployments.

---

### 🔄 Step 4: CI/CD (Optional Enhancements)
- Use **Cloud Build** or **GitHub Actions** for continuous integration.
- Use **Kustomize** or **Helm** for deployment templating.

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

With this project, you're not just deploying a Flask app. You're deploying **an intelligent observability engine** that watches over your Kubernetes infrastructure like a hawk. Integrated with GKE, it's scalable, resilient, and production-ready.



## Dataset :

[Azure Public Dataset - VM Traces](github.com/Azure/AzurePublicDataset)

```
Source: https://github.com/Azure/AzurePublicDataset
Description: Contains VM workload traces that can be translated to Kubernetes workloads
Advantage: Well-documented and includes CPU, memory usage patterns
```

## Developers 


- [@Sai Nivedh V](https://github.com/SaiNivedh26)
- [@Baranidharan S](https://github.com/thespectacular314)
- [@Roshan T](https://github.com/Twinn-github09)
- [@Hari Heman](https://github.com/MAD-MAN-HEMAN)
- [@Kavinesh](https://github.com/Kavinesh11)
