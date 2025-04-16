from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import os
import time
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('prediction_service')

app = Flask(__name__)

# Load the models and feature information
def load_models():
    try:
        binary_model = joblib.load('models/binary_classifier.pkl')
        multiclass_model = joblib.load('models/multiclass_classifier.pkl')
        feature_info = joblib.load('models/feature_info.pkl')
        return binary_model, multiclass_model, feature_info
    except FileNotFoundError:
        logger.error("Model files not found. Please train the models first.")
        return None, None, None

binary_model, multiclass_model, feature_info = load_models()

# Dictionary to store simulated cluster data
cluster_data = {}

def generate_cluster_metrics(cluster_id, timestamp=None):
    """
    Generate simulated cluster metrics for testing or development
    
    In production, this would be replaced by actual metrics from
    the Kubernetes API or monitoring system
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    # Check if we have previous data for this cluster
    if cluster_id in cluster_data:
        prev_metrics = cluster_data[cluster_id]
        
        # Slightly change metrics based on previous values
        base_cpu_util = prev_metrics.get('cluster_cpu_util_avg', 50)
        # Add some random change (-5 to +5)
        cpu_change = np.random.normal(0, 2)
        cpu_util = max(min(base_cpu_util + cpu_change, 100), 0)
        
        # Initialize node_count with a default value
        node_count = prev_metrics.get('node_count', 5)
        
        # Occasionally generate anomalous values
        if np.random.random() < 0.05:  # 5% chance of anomaly
            anomaly_type = np.random.choice(['cpu_spike', 'node_drop', 'normal'])
            if anomaly_type == 'cpu_spike':
                cpu_util = min(cpu_util + np.random.uniform(20, 40), 100)
            elif anomaly_type == 'node_drop':
                node_count = max(node_count - 1, 1)
            # For 'normal' anomaly type, node_count remains unchanged
        
        # Calculate CPU standard deviation
        cpu_std = prev_metrics.get('cluster_cpu_util_std', 10)
        cpu_std = max(min(cpu_std + np.random.normal(0, 1), 20), 1)
        
    else:
        # First data point for this cluster
        cpu_util = np.random.uniform(30, 70)
        node_count = np.random.randint(3, 10)
        cpu_std = np.random.uniform(5, 15)
    
    # Generate all required metrics
    metrics = {
        'timestamp': timestamp,
        'cluster_id': cluster_id,
        'cluster_cpu_util_avg': cpu_util,
        'cluster_cpu_util_max': min(cpu_util + cpu_std * 1.5, 100),
        'cluster_cpu_util_min': max(cpu_util - cpu_std * 1.5, 0),
        'cluster_cpu_util_std': cpu_std,
        'node_count': node_count,
        'sample_count': node_count * np.random.randint(1, 5),
        'prev_node_count': cluster_data.get(cluster_id, {}).get('node_count', node_count),
        'node_count_change': node_count - cluster_data.get(cluster_id, {}).get('node_count', node_count),
        'prev_cpu_avg': cluster_data.get(cluster_id, {}).get('cluster_cpu_util_avg', cpu_util),
        'cpu_change_rate': cpu_util - cluster_data.get(cluster_id, {}).get('cluster_cpu_util_avg', cpu_util),
        'hour': timestamp.hour,
        'day_of_week': timestamp.weekday(),
        'is_business_hours': 1 if 9 <= timestamp.hour <= 17 and timestamp.weekday() < 5 else 0,
        'vmcategory': np.random.choice(['WebServer', 'Database', 'Worker']),
    }
    
    # Calculate rolling features if we have history
    if cluster_id in cluster_data:
        metrics['cpu_avg_1h'] = (cluster_data[cluster_id].get('cpu_avg_1h', cpu_util) * 0.8 + 
                                cpu_util * 0.2)
        metrics['cpu_max_1h'] = max(cluster_data[cluster_id].get('cpu_max_1h', cpu_util), 
                                   metrics['cluster_cpu_util_max'])
        metrics['cpu_std_1h'] = (cluster_data[cluster_id].get('cpu_std_1h', cpu_std) * 0.8 + 
                                cpu_std * 0.2)
        metrics['node_count_change_1h'] = node_count - cluster_data[cluster_id].get('node_count_1h_ago', node_count)
        
        # Update the 1-hour-ago node count
        if 'timestamp' in cluster_data[cluster_id]:
            prev_time = cluster_data[cluster_id]['timestamp']
            if (timestamp - prev_time).total_seconds() >= 3600:  # 1 hour
                cluster_data[cluster_id]['node_count_1h_ago'] = node_count
    else:
        metrics['cpu_avg_1h'] = cpu_util
        metrics['cpu_max_1h'] = metrics['cluster_cpu_util_max']
        metrics['cpu_std_1h'] = cpu_std
        metrics['node_count_change_1h'] = 0
        cluster_data[cluster_id] = {'node_count_1h_ago': node_count}
    
    # Store current metrics for next time
    cluster_data[cluster_id] = metrics
    
    return metrics

def predict_issues(cluster_metrics):
    """Make predictions using the loaded models"""
    if binary_model is None or multiclass_model is None:
        return None, None
    
    # Extract features needed by the models
    numerical_features = feature_info['numerical_features']
    categorical_features = feature_info['categorical_features']
    
    # Create DataFrame with required features
    features_df = pd.DataFrame([cluster_metrics])
    
    # Check if the required features exist in the data
    missing_features = []
    for feature in numerical_features + categorical_features:
        if feature not in features_df.columns:
            # Look for similar column names that might match
            possible_match = [col for col in features_df.columns if feature.replace('_<lambda>', '') in col]
            if possible_match:
                # Rename the column to match what the model expects
                features_df[feature] = features_df[possible_match[0]]
            else:
                missing_features.append(feature)
                # Add a placeholder value (0) to avoid errors
                features_df[feature] = 0
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
    
    # Select only the features needed
    X = features_df[numerical_features + categorical_features]
    
    # Make predictions
    issue_probability = binary_model.predict_proba(X)[0, 1]
    will_have_issue = binary_model.predict(X)[0]
    
    issue_type = None
    if will_have_issue:
        issue_type = multiclass_model.predict(X)[0]
    
    return will_have_issue, issue_probability, issue_type

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive metrics and return predictions
    
    In production, this would receive actual metrics from Kubernetes.
    For testing, we'll accept simulated metrics or generate them.
    """
    # Check if we received metrics
    if request.is_json:
        data = request.get_json()
        # Use provided metrics
        cluster_metrics = data
    else:
        # Generate simulated metrics for testing
        cluster_id = request.args.get('cluster_id', '1')
        cluster_metrics = generate_cluster_metrics(cluster_id)
    
    # Make predictions
    will_have_issue, probability, issue_type = predict_issues(cluster_metrics)
    
    # In case models aren't loaded
    if will_have_issue is None:
        return jsonify({
            'error': 'Models not loaded. Please train the models first.'
        }), 500
    
    response = {
        'timestamp': datetime.now().isoformat(),
        'cluster_id': cluster_metrics['cluster_id'],
        'prediction': {
            'will_have_issue': bool(will_have_issue),
            'issue_probability': float(probability),
            'issue_type': issue_type if will_have_issue else 'normal',
            'recommendation': get_recommendation(issue_type) if will_have_issue else None
        },
        'metrics': {
            'cpu_utilization': cluster_metrics['cluster_cpu_util_avg'],
            'node_count': cluster_metrics['node_count']
        }
    }
    
    return jsonify(response)

@app.route('/simulate/<int:count>', methods=['GET'])
def simulate_clusters(count):
    """
    Generate simulated data for multiple clusters and make predictions
    Useful for testing and demonstrations
    """
    results = []
    timestamp = datetime.now()
    
    for i in range(1, count + 1):
        cluster_id = str(i)
        metrics = generate_cluster_metrics(cluster_id, timestamp)
        will_have_issue, probability, issue_type = predict_issues(metrics)
        
        # Only include clusters with potential issues in the response
        if probability > 0.3:  # Include moderate to high probability issues
            results.append({
                'cluster_id': cluster_id,
                'prediction': {
                    'will_have_issue': bool(will_have_issue),
                    'issue_probability': float(probability),
                    'issue_type': issue_type if will_have_issue else 'normal',
                    'recommendation': get_recommendation(issue_type) if will_have_issue else None
                },
                'metrics': {
                    'cpu_utilization': metrics['cluster_cpu_util_avg'],
                    'cpu_max': metrics['cluster_cpu_util_max'],
                    'node_count': metrics['node_count']
                }
            })
    
    return jsonify({
        'timestamp': timestamp.isoformat(),
        'clusters': results
    })

def get_recommendation(issue_type):
    """Generate remediation recommendations based on the predicted issue type"""
    recommendations = {
        'resource_exhaustion': {
            'description': 'The cluster is experiencing high resource utilization',
            'actions': [
                'Scale up the affected deployments',
                'Add more nodes to the cluster',
                'Implement resource quotas to prevent overutilization'
            ]
        },
        'pod_failure': {
            'description': 'Pods are failing or being terminated unexpectedly',
            'actions': [
                'Investigate pod logs for error messages',
                'Check for memory leaks or OOM kills',
                'Restart affected deployments',
                'Consider implementing liveness and readiness probes'
            ]
        },
        'network_issue': {
            'description': 'Network connectivity issues detected',
            'actions': [
                'Check network policies and security groups',
                'Verify DNS resolution within the cluster',
                'Investigate service mesh configurations',
                'Check for network saturation or throttling'
            ]
        }
    }
    
    return recommendations.get(issue_type, {
        'description': f'Unknown issue type: {issue_type}',
        'actions': ['Investigate cluster logs', 'Monitor cluster metrics']
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    status = 'healthy' if binary_model is not None and multiclass_model is not None else 'models_not_loaded'
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)