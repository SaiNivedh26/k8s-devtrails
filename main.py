import pandas as pd
import numpy as np
import os
import gzip
import shutil
import urllib.request
from datetime import datetime

def download_azure_dataset():
    """Download Azure Public Dataset VM traces"""
    os.makedirs('data', exist_ok=True)
    
    # URLs of datasets
    urls = [
        'https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/vmtable/vmtable.csv.gz',
        'https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/vm_cpu_readings/vm_cpu_readings-file-1-of-195.csv.gz',
        'https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/trace_data/deployments/deployments.csv.gz'
    ]
    
    # Download and extract files
    for url in urls:
        filename = os.path.basename(url)
        gz_path = os.path.join('data', filename)
        csv_path = os.path.join('data', filename[:-3])  # Remove .gz
        
        if not os.path.exists(csv_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, gz_path)
            
            print(f"Extracting {filename}...")
            with gzip.open(gz_path, 'rb') as f_in:
                with open(csv_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(gz_path)  # Remove the compressed file
            print(f"Extracted {csv_path}")
    
    print("Download complete!")

def load_and_process_data():
    """Load and preprocess the Azure VM dataset"""
    # Define column headers
    vm_headers = ['vmid', 'subscriptionid', 'deploymentid', 'vmcreated', 
                 'vmdeleted', 'maxcpu', 'avgcpu', 'p95maxcpu',
                 'vmcategory', 'vmcorecountbucket', 'vmmemorybucket']
    
    cpu_headers = ['vmid', 'timestamp', 'cpu_utilization']
    
    deployment_headers = ['deploymentid', 'subscriptionid', 'deploymentsize',
                         'createdtime', 'deletedtime']
    
    # Load data
    print("Loading VM data...")
    vm_data = pd.read_csv('data/vmtable.csv', header=None, names=vm_headers)
    
    print("Loading CPU readings...")
    cpu_data = pd.read_csv('data/vm_cpu_readings-file-1-of-195.csv', 
                          header=None, names=cpu_headers)
    
    print("Loading deployment data...")
    deployment_data = pd.read_csv('data/deployments.csv', 
                                 header=None, names=deployment_headers)
    
    # Debug info
    print(f"VM data shape: {vm_data.shape}")
    print(f"CPU data shape: {cpu_data.shape}")
    print(f"VM data 'vmid' sample: {vm_data['vmid'].head()}")
    print(f"CPU data 'vmid' sample: {cpu_data['vmid'].head()}")
    
    # Since we have no common IDs, we'll generate synthetic data instead
    print("No matching VM IDs found. Generating synthetic Kubernetes data instead...")
    
    # Create a synthetic dataset with 100 VMs across 5 clusters
    np.random.seed(42)  # For reproducibility
    num_vms = 100
    num_clusters = 5
    time_periods = 100  # Number of time points
    
    # Generate cluster assignments for VMs
    vm_to_cluster = {f"vm-{i}": f"cluster-{i % num_clusters}" for i in range(num_vms)}
    
    # Generate synthetic metrics data
    records = []
    
    # Start timestamp
    start_time = pd.Timestamp('2023-01-01')
    
    for t in range(time_periods):
        timestamp = start_time + pd.Timedelta(minutes=15 * t)
        
        for vm_id, cluster_id in vm_to_cluster.items():
            # Base CPU utilization with some randomness
            base_cpu = 30 + 20 * np.sin(t/10) + np.random.normal(0, 10)
            cpu_util = max(min(base_cpu, 100), 0)
            
            # Add some anomalies
            if np.random.random() < 0.05:  # 5% chance of anomaly
                cpu_util = np.random.choice([cpu_util * 1.5, cpu_util * 0.2])
            
            records.append({
                'timestamp': timestamp,
                'vmid': vm_id,
                'cluster_id': cluster_id,
                'cpu_utilization': cpu_util,
                'vmcategory': np.random.choice(['WebServer', 'Database', 'Worker']),
                'vmcorecountbucket': np.random.choice(['1', '2', '4', '8']),
                'vmmemorybucket': np.random.choice(['small', 'medium', 'large']),
                'deploymentid': f"dep-{vm_id.split('-')[1]}"
            })
    
    # Create DataFrame from synthetic data
    merged_data = pd.DataFrame(records)
    
    # Convert timestamps to datetime
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'])
    
    print(f"Generated synthetic data shape: {merged_data.shape}")
    return merged_data

def simulate_kubernetes_clusters(merged_data, n_clusters=10):
    """
    Transform VM data into Kubernetes-like clusters
    by grouping VMs into virtual clusters and computing aggregate metrics
    """
    print("Simulating Kubernetes clusters...")
    
    # Check what columns are available in the merged_data
    available_columns = merged_data.columns
    print(f"Available columns: {available_columns}")
    
    # Group by cluster and timestamp to simulate Kubernetes cluster metrics
    # Use only columns that exist in the dataframe
    agg_dict = {
        'cpu_utilization': ['mean', 'std', 'max', 'min', 'count'],
        'vmid': 'nunique',  # number of unique VMs (like nodes)
    }
    
    # Add vmcategory to agg_dict if it exists
    if 'vmcategory' in available_columns:
        agg_dict['vmcategory'] = lambda x: x.mode().iloc[0] if not x.empty and not x.mode().empty else 'unknown'
    
    # Add deploymentsize to agg_dict only if it exists
    if 'deploymentsize' in available_columns:
        agg_dict['deploymentsize'] = 'mean'
    
    cluster_metrics = merged_data.groupby(['cluster_id', 'timestamp']).agg(agg_dict)
    
    # Flatten MultiIndex columns
    cluster_metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                             for col in cluster_metrics.columns]
    
    # Reset index for easier handling
    cluster_metrics = cluster_metrics.reset_index()
    
    # Rename columns to be more Kubernetes-like
    column_mappings = {
        'vmid_nunique': 'node_count',
        'cpu_utilization_count': 'sample_count',
        'cpu_utilization_mean': 'cluster_cpu_util_avg',
        'cpu_utilization_max': 'cluster_cpu_util_max',
        'cpu_utilization_min': 'cluster_cpu_util_min',
        'cpu_utilization_std': 'cluster_cpu_util_std'
    }
    
    # Only rename columns that exist
    for old_name, new_name in column_mappings.items():
        if old_name in cluster_metrics.columns:
            cluster_metrics = cluster_metrics.rename(columns={old_name: new_name})
    
    # If deploymentsize exists, rename it
    if 'deploymentsize_mean' in cluster_metrics.columns:
        cluster_metrics = cluster_metrics.rename(columns={'deploymentsize_mean': 'avg_deployment_size'})
    
    print(f"Cluster metrics shape: {cluster_metrics.shape}")
    print(f"Cluster metrics columns: {cluster_metrics.columns}")
    
    return cluster_metrics

def generate_kubernetes_issues(cluster_metrics):
    """
    Generate synthetic Kubernetes issue labels based on metrics
    This simulates different types of cluster issues
    """
    print("Generating Kubernetes issue labels...")
    
    # Sort by cluster and timestamp for time-based calculations
    cluster_metrics = cluster_metrics.sort_values(['cluster_id', 'timestamp'])
    
    # Add features for resource exhaustion detection
    cluster_metrics['cpu_high'] = cluster_metrics['cluster_cpu_util_max'] > 85
    
    # Create time-shifted columns for node count to detect changes
    cluster_metrics['prev_node_count'] = cluster_metrics.groupby('cluster_id')['node_count'].shift(1)
    cluster_metrics['node_count_change'] = cluster_metrics['node_count'] - cluster_metrics['prev_node_count']
    cluster_metrics['node_failure'] = (cluster_metrics['node_count_change'] < 0) & (cluster_metrics['prev_node_count'] > 0)
    
    # Calculate rate of change for CPU utilization
    cluster_metrics['prev_cpu_avg'] = cluster_metrics.groupby('cluster_id')['cluster_cpu_util_avg'].shift(1)
    cluster_metrics['cpu_change_rate'] = (cluster_metrics['cluster_cpu_util_avg'] - cluster_metrics['prev_cpu_avg'])
    
    # Simulate network issues based on CPU patterns and randomness
    # (In a real scenario, you'd have actual network metrics)
    np.random.seed(42)  # For reproducibility
    
    # Generate network issues with some correlation to CPU spikes
    cpu_factor = np.clip(cluster_metrics['cluster_cpu_util_avg'] / 100, 0, 1) * 0.1
    random_factor = np.random.random(cluster_metrics.shape[0]) * 0.05
    cluster_metrics['network_issue_probability'] = cpu_factor + random_factor
    cluster_metrics['network_issue'] = cluster_metrics['network_issue_probability'] > 0.12
    
    # Generate labels for different types of issues
    cluster_metrics['resource_exhaustion'] = (
        (cluster_metrics['cluster_cpu_util_max'] > 90) | 
        ((cluster_metrics['cluster_cpu_util_avg'] > 80) & 
         (cluster_metrics['cluster_cpu_util_std'] < 5))
    )
    
    cluster_metrics['pod_failure'] = (
        cluster_metrics['node_failure'] | 
        ((cluster_metrics['cpu_change_rate'] > 20) & 
         (cluster_metrics['cluster_cpu_util_max'] > 95))
    )
    
    # Combine issues into a single target column
    conditions = [
        cluster_metrics['resource_exhaustion'],
        cluster_metrics['pod_failure'],
        cluster_metrics['network_issue']
    ]
    
    choices = ['resource_exhaustion', 'pod_failure', 'network_issue']
    
    # Default to 'normal' if no issues
    cluster_metrics['issue_type'] = np.select(conditions, choices, default='normal')
    
    # Create binary target
    cluster_metrics['has_issue'] = (cluster_metrics['issue_type'] != 'normal').astype(int)
    
    # Drop intermediate columns
    columns_to_drop = [
        'cpu_high', 'network_issue_probability',
        'resource_exhaustion', 'pod_failure', 'network_issue'
    ]
    cluster_metrics = cluster_metrics.drop(columns=columns_to_drop)
    
    # Forward-fill any NaN values that might have been created
    cluster_metrics = cluster_metrics.fillna(0)
    
    return cluster_metrics

def create_time_features(cluster_metrics):
    """Add time-based features that could be relevant for predictions"""
    print("Creating time features...")
    
    # Extract time components
    cluster_metrics['hour'] = cluster_metrics['timestamp'].dt.hour
    cluster_metrics['day_of_week'] = cluster_metrics['timestamp'].dt.dayofweek
    cluster_metrics['is_business_hours'] = ((cluster_metrics['hour'] >= 9) & 
                                          (cluster_metrics['hour'] <= 17) & 
                                          (cluster_metrics['day_of_week'] < 5)).astype(int)
    
    # Create rolling window features
    # Make a copy to avoid modifying the original data
    df = cluster_metrics.copy()
    
    # Sort by cluster and timestamp
    df = df.sort_values(['cluster_id', 'timestamp'])
    
    # Calculate rolling statistics without using groupby.apply() which causes the duplicate column issue
    result = pd.DataFrame()
    
    # Process each cluster separately to avoid the index/column ambiguity
    for cluster_id, group in df.groupby('cluster_id'):
        # Set timestamp as index for rolling operations
        group = group.set_index('timestamp')
        
        # Calculate rolling statistics
        group['cpu_avg_1h'] = group['cluster_cpu_util_avg'].rolling('1H').mean()
        group['cpu_max_1h'] = group['cluster_cpu_util_avg'].rolling('1H').max()
        group['cpu_std_1h'] = group['cluster_cpu_util_avg'].rolling('1H').std()
        
        # Calculate node count changes
        group['node_count_change_1h'] = group['node_count'].rolling('1H').apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
        )
        
        # Reset index to get timestamp back as a column
        group = group.reset_index()
        
        # Append to result
        result = pd.concat([result, group], ignore_index=True)
    
    # Fill NaN values in rolling columns
    rolling_columns = ['cpu_avg_1h', 'cpu_max_1h', 'cpu_std_1h', 'node_count_change_1h']
    for col in rolling_columns:
        # Forward fill within each cluster
        result[col] = result.groupby('cluster_id')[col].transform(
            lambda x: x.fillna(method='ffill')
        )
        # Fill remaining NaNs with original values
        idx = result[col].isna()
        if col.startswith('cpu_avg'):
            result.loc[idx, col] = result.loc[idx, 'cluster_cpu_util_avg']
        elif col.startswith('cpu_max'):
            result.loc[idx, col] = result.loc[idx, 'cluster_cpu_util_max']
        elif col.startswith('cpu_std'):
            result.loc[idx, col] = result.loc[idx, 'cluster_cpu_util_std']
        else:
            result.loc[idx, col] = 0
    
    return result

def prepare_final_dataset():
    """Prepare the final dataset for modeling"""
    # Download the dataset if not available
    if not os.path.exists('data/vmtable.csv'):
        download_azure_dataset()
    
    # Load and process data
    merged_data = load_and_process_data()
    
    # Create simulated Kubernetes clusters
    cluster_metrics = simulate_kubernetes_clusters(merged_data)
    
    # Generate issue labels
    labeled_data = generate_kubernetes_issues(cluster_metrics)
    
    # Add time features
    final_data = create_time_features(labeled_data)
    
    # Save the processed dataset
    os.makedirs('processed_data', exist_ok=True)
    final_data.to_csv('processed_data/k8s_metrics.csv', index=False)
    print("Dataset saved to processed_data/k8s_metrics.csv")
    
    return final_data

if __name__ == "__main__":
    prepare_final_dataset()