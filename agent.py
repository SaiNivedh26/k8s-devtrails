import requests
import time
import logging
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import kubernetes.client
from kubernetes.client.rest import ApiException
from kubernetes import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('remediation.log')]
)
logger = logging.getLogger('remediation_agent')

class RemediationAgent:
    def __init__(self, prediction_service_url='http://localhost:5000'):
        """
        Initialize the remediation agent
        
        Args:
            prediction_service_url: URL of the prediction service API
        """
        self.prediction_service_url = prediction_service_url
        self.last_actions = {}  # Store when actions were last taken
        self.action_cooldown = 300  # seconds (5 minutes)
        
        # Connect to Kubernetes API
        try:
            # Try in-cluster config first
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except config.ConfigException:
            # Fall back to kubeconfig
            try:
                config.load_kube_config()
                logger.info("Using kubeconfig for Kubernetes configuration")
            except config.ConfigException:
                logger.warning("Could not load Kubernetes configuration. Running in simulation mode.")
        
        # Initialize Kubernetes API clients
        self.core_api = kubernetes.client.CoreV1Api()
        self.apps_api = kubernetes.client.AppsV1Api()
        self.autoscaling_api = kubernetes.client.AutoscalingV1Api()
        
        # Simulation mode - in case we're not running in a real Kubernetes cluster
        self.simulation_mode = os.environ.get('SIMULATION_MODE', 'false').lower() == 'true'
        if self.simulation_mode:
            logger.info("Running in simulation mode. Kubernetes actions will be simulated.")
        
        # Create record of remediation actions
        self.remediation_history = []
    
    def get_predictions(self, cluster_count=5):
        """Get predictions from the prediction service"""
        try:
            # For testing, we'll use the simulation endpoint
            response = requests.get(f"{self.prediction_service_url}/simulate/{cluster_count}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting predictions: {e}")
            return None
    
    def evaluate_and_remediate(self):
        """Evaluate predictions and take remediation actions as needed"""
        # Get predictions
        predictions = self.get_predictions()
        if not predictions:
            logger.error("No predictions received")
            return False
        
        # Process each cluster
        for cluster in predictions.get('clusters', []):
            cluster_id = cluster['cluster_id']
            prediction = cluster['prediction']
            metrics = cluster['metrics']
            
            # Check if remediation is needed
            if prediction['will_have_issue']:
                issue_type = prediction['issue_type']
                probability = prediction['issue_probability']
                
                logger.info(f"Cluster {cluster_id} predicted to have {issue_type} issue " +
                           f"(probability: {probability:.2f})")
                
                # Only take action if probability is high enough
                if probability > 0.7:  # 70% threshold
                    # Check cooldown period
                    last_action_time = self.last_actions.get(f"{cluster_id}_{issue_type}")
                    current_time = time.time()
                    
                    if (last_action_time is None or 
                        current_time - last_action_time > self.action_cooldown):
                        
                        # Take remediation action based on issue type
                        success = self.take_remediation_action(cluster_id, issue_type, metrics)
                        
                        if success:
                            # Update last action time
                            self.last_actions[f"{cluster_id}_{issue_type}"] = current_time
                    else:
                        cooldown_remaining = int(self.action_cooldown - (current_time - last_action_time))
                        logger.info(f"Skipping remediation for cluster {cluster_id} due to " +
                                   f"cooldown period ({cooldown_remaining}s remaining)")
        
        return True
    
    def take_remediation_action(self, cluster_id, issue_type, metrics):
        """Take appropriate remediation action based on the issue type"""
        action_taken = False
        action_details = {}
        
        # Common attributes for logging
        action_record = {
            'timestamp': datetime.now().isoformat(),
            'cluster_id': cluster_id,
            'issue_type': issue_type,
            'metrics': metrics
        }
        
        try:
            if issue_type == 'resource_exhaustion':
                # Handle resource exhaustion
                action_taken, action_details = self.handle_resource_exhaustion(cluster_id, metrics)
            elif issue_type == 'pod_failure':
                # Handle pod failures
                action_taken, action_details = self.handle_pod_failure(cluster_id)
            elif issue_type == 'network_issue':
                # Handle network issues
                action_taken, action_details = self.handle_network_issue(cluster_id)
            else:
                logger.warning(f"Unknown issue type: {issue_type}")
                return False
            
            # Record the action
            action_record['action_taken'] = action_taken
            action_record['action_details'] = action_details
            self.remediation_history.append(action_record)
            
            # Save remediation history periodically
            if len(self.remediation_history) % 10 == 0:
                self.save_remediation_history()
            
            return action_taken
        
        except Exception as e:
            logger.error(f"Error during remediation for cluster {cluster_id}: {e}")
            action_record['error'] = str(e)
            self.remediation_history.append(action_record)
            return False
    
    def handle_resource_exhaustion(self, cluster_id, metrics):
        """
        Handle resource exhaustion issues
        
        This could involve:
        - Scaling up deployments
        - Adding nodes to the cluster
        - Optimizing resource allocation
        """
        logger.info(f"Handling resource exhaustion for cluster {cluster_id}")
        
        action_details = {'cpu_utilization': metrics['cpu_utilization']}
        
        if self.simulation_mode:
            # Simulate scaling action
            logger.info(f"SIMULATION: Scaling up deployments in cluster {cluster_id}")
            action_details['action'] = 'simulate_scale_up'
            return True, action_details
        
        try:
            # Get deployments with high CPU usage
            deployments = self.apps_api.list_deployment_for_all_namespaces()
            scaled_deployments = []
            
            for deployment in deployments.items:
                # In a real scenario, we would check metrics for each deployment
                # Here we'll randomly select some to scale
                if np.random.random() < 0.3:  # Simulate finding deployments that need scaling
                    try:
                        name = deployment.metadata.name
                        namespace = deployment.metadata.namespace
                        current_replicas = deployment.spec.replicas
                        
                        # Only scale if replica count is reasonable
                        if current_replicas and current_replicas < 10:
                            # Scale up by 1 or 2 replicas
                            new_replicas = current_replicas + max(1, int(current_replicas * 0.2))
                            
                            # Update deployment
                            deployment.spec.replicas = new_replicas
                            self.apps_api.patch_namespaced_deployment(
                                name=name,
                                namespace=namespace,
                                body=deployment
                            )
                            
                            logger.info(f"Scaled deployment {name} in namespace {namespace} " +
                                      f"from {current_replicas} to {new_replicas} replicas")
                            
                            scaled_deployments.append({
                                'name': name,
                                'namespace': namespace,
                                'old_replicas': current_replicas,
                                'new_replicas': new_replicas
                            })
                    except ApiException as e:
                        logger.error(f"Error scaling deployment {name}: {e}")
            
            action_details['scaled_deployments'] = scaled_deployments
            return len(scaled_deployments) > 0, action_details
        
        except ApiException as e:
            logger.error(f"Error accessing Kubernetes API: {e}")
            action_details['error'] = str(e)
            return False, action_details
    
    def handle_pod_failure(self, cluster_id):
        """
        Handle pod failures
        
        This could involve:
        - Restarting failing pods
        - Checking for resource constraints
        - Relocating pods to different nodes
        """
        logger.info(f"Handling pod failures for cluster {cluster_id}")
        
        action_details = {}
        
        if self.simulation_mode:
            # Simulate pod restart action
            logger.info(f"SIMULATION: Restarting problematic pods in cluster {cluster_id}")
            action_details['action'] = 'simulate_pod_restart'
            return True, action_details
        
        try:
            # Find pods with high restart counts or in bad states
            pods = self.core_api.list_pod_for_all_namespaces()
            problematic_pods = []
            
            for pod in pods.items:
                is_problematic = False
                restart_counts = []
                
                # Check container statuses
                if pod.status.container_statuses:
                    for container in pod.status.container_statuses:
                        # Check for containers with many restarts
                        if container.restart_count > 5:
                            is_problematic = True
                            restart_counts.append(container.restart_count)
                        
                        # Check for containers in CrashLoopBackOff
                        if (container.state.waiting and 
                            container.state.waiting.reason == 'CrashLoopBackOff'):
                            is_problematic = True
                
                # Check if pod is in a problematic phase
                if pod.status.phase in ['Failed', 'Unknown']:
                    is_problematic = True
                
                if is_problematic:
                    name = pod.metadata.name
                    namespace = pod.metadata.namespace
                    
                    # Delete the pod to force a restart
                    try:
                        self.core_api.delete_namespaced_pod(name=name, namespace=namespace)
                        logger.info(f"Restarted problematic pod {name} in namespace {namespace}")
                        
                        problematic_pods.append({
                            'name': name,
                            'namespace': namespace,
                            'phase': pod.status.phase,
                            'restart_counts': restart_counts
                        })
                    except ApiException as e:
                        logger.error(f"Error deleting pod {name}: {e}")
            
            action_details['restarted_pods'] = problematic_pods
            return len(problematic_pods) > 0, action_details
        
        except ApiException as e:
            logger.error(f"Error accessing Kubernetes API: {e}")
            action_details['error'] = str(e)
            return False, action_details
    
    def handle_network_issue(self, cluster_id):
        """
        Handle network connectivity issues
        
        This could involve:
        - Checking network policies
        - Restarting networking components
        - Verifying service mesh configuration
        """
        logger.info(f"Handling network issues for cluster {cluster_id}")
        
        action_details = {}
        
        if self.simulation_mode:
            # Simulate network fix action
            logger.info(f"SIMULATION: Taking actions to address network issues in cluster {cluster_id}")
            action_details['action'] = 'simulate_network_fix'
            return True, action_details
        
        try:
            # In a real implementation, we might:
            # 1. Restart CoreDNS pods
            # 2. Validate NetworkPolicies
            # 3. Check node networking
            
            # For this example, we'll restart CoreDNS pods which often helps with DNS issues
            coredns_pods = self.core_api.list_namespaced_pod(
                namespace="kube-system", 
                label_selector="k8s-app=kube-dns"
            )
            
            restarted_pods = []
            
            for pod in coredns_pods.items:
                name = pod.metadata.name
                namespace = pod.metadata.namespace
                
                # Delete the pod to force a restart
                try:
                    self.core_api.delete_namespaced_pod(name=name, namespace=namespace)
                    logger.info(f"Restarted CoreDNS pod {name}")
                    
                    restarted_pods.append({
                        'name': name,
                        'namespace': namespace
                    })
                except ApiException as e:
                    logger.error(f"Error deleting CoreDNS pod {name}: {e}")
            
            # Also check for networking components like CNI
            network_pods = self.core_api.list_namespaced_pod(
                namespace="kube-system", 
                label_selector="k8s-app=calico-node,k8s-app=flannel"
            )
            
            for pod in network_pods.items:
                if pod.status.phase != 'Running':
                    name = pod.metadata.name
                    namespace = pod.metadata.namespace
                    
                    # Delete to force restart
                    try:
                        self.core_api.delete_namespaced_pod(name=name, namespace=namespace)
                        logger.info(f"Restarted networking pod {name}")
                        
                        restarted_pods.append({
                            'name': name,
                            'namespace': namespace
                        })
                    except ApiException as e:
                        logger.error(f"Error deleting networking pod {name}: {e}")
            
            action_details['restarted_pods'] = restarted_pods
            return len(restarted_pods) > 0, action_details
        
        except ApiException as e:
            logger.error(f"Error accessing Kubernetes API: {e}")
            action_details['error'] = str(e)
            return False, action_details
    
    def save_remediation_history(self):
        """Save remediation history to a file"""
        try:
            os.makedirs('logs', exist_ok=True)
            
            with open('logs/remediation_history.json', 'w') as f:
                json.dump(self.remediation_history, f, indent=2)
            
            logger.debug("Remediation history saved")
        except Exception as e:
            logger.error(f"Error saving remediation history: {e}")
    
    def run(self, interval=60):
        """Run the remediation agent in a loop"""
        logger.info(f"Starting remediation agent, checking every {interval} seconds")
        
        try:
            while True:
                # Evaluate predictions and take actions
                self.evaluate_and_remediate()
                
                # Sleep until next check
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Remediation agent stopped")
            self.save_remediation_history()

if __name__ == "__main__":
    # Set simulation mode via environment variable, defaults to true for safety
    simulation_mode = os.environ.get('SIMULATION_MODE', 'true').lower() == 'true'
    
    # Set prediction service URL
    prediction_url = os.environ.get('PREDICTION_SERVICE_URL', 'http://localhost:5000')
    
    # Create and run the remediation agent
    agent = RemediationAgent(prediction_service_url=prediction_url)
    agent.run()