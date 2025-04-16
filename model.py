import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed dataset"""
    data_path = 'processed_data/k8s_metrics.csv'
    
    if not os.path.exists(data_path):
        from main import prepare_final_dataset
        data = prepare_final_dataset()
    else:
        data = pd.read_csv(data_path)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    return data

def visualize_data(data):
    """Create visualizations to understand the data"""
    os.makedirs('plots', exist_ok=True)
    
    # 1. Distribution of CPU utilization - use plt.hist instead of seaborn
    plt.figure(figsize=(10, 6))
    plt.hist(data['cluster_cpu_util_avg'].values, bins=30, alpha=0.7, density=True)
    plt.title('Distribution of Cluster CPU Utilization')
    plt.xlabel('CPU Utilization (%)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig('plots/cpu_distribution.png')
    plt.close()
    
    # 2. Issue distribution - use plt.bar instead of seaborn
    plt.figure(figsize=(8, 6))
    issue_counts = data['issue_type'].value_counts()
    plt.bar(issue_counts.index, issue_counts.values)
    plt.title('Distribution of Issue Types')
    plt.xlabel('Issue Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/issue_distribution.png')
    plt.close()
    
    # 3. CPU utilization by issue type - use matplotlib directly
    plt.figure(figsize=(10, 6))
    issue_types = data['issue_type'].unique()
    box_data = [data[data['issue_type'] == issue]['cluster_cpu_util_avg'].values for issue in issue_types]
    plt.boxplot(box_data, labels=issue_types)
    plt.title('CPU Utilization by Issue Type')
    plt.xlabel('Issue Type')
    plt.ylabel('CPU Utilization (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/cpu_by_issue.png')
    plt.close()
    
    # 4. Time series of CPU utilization for a sample cluster
    cluster_ids = data['cluster_id'].unique()
    if len(cluster_ids) > 0:
        sample_cluster_id = cluster_ids[0]
        sample_cluster = data[data['cluster_id'] == sample_cluster_id]
        plt.figure(figsize=(14, 7))
        
        # Convert to numpy arrays explicitly
        times = sample_cluster['timestamp'].values
        cpu_values = sample_cluster['cluster_cpu_util_avg'].values
        
        plt.plot(times, cpu_values, '-o', markersize=4)
        
        # Highlight issues
        issues = sample_cluster[sample_cluster['has_issue'] == 1]
        if len(issues) > 0:
            issue_times = issues['timestamp'].values
            issue_cpu_values = issues['cluster_cpu_util_avg'].values
            plt.scatter(issue_times, issue_cpu_values, color='red', s=50, label='Issue')
            plt.legend()
        
        plt.title(f'CPU Utilization Over Time for Cluster {sample_cluster_id}')
        plt.xlabel('Time')
        plt.ylabel('CPU Utilization (%)')
        plt.tight_layout()
        plt.savefig('plots/time_series_cpu.png')
        plt.close()

def prepare_features(data, predict_ahead_minutes=30):
    """
    Prepare features for model training
    Also shifts the target to allow prediction ahead of time
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # For prediction ahead of time, we need to shift the target
    # First, sort data by cluster and time
    df = df.sort_values(['cluster_id', 'timestamp'])
    
    # Calculate the time difference in minutes between consecutive records
    df['time_diff'] = df.groupby('cluster_id')['timestamp'].diff().dt.total_seconds() / 60
    
    # Find the typical time interval in the data
    median_interval = df['time_diff'].median()
    
    # Calculate how many steps ahead to shift based on the desired prediction time
    steps_ahead = int(round(predict_ahead_minutes / median_interval))
    
    # Shift the target variables ahead by the calculated steps
    # This means the features at time t will be used to predict issues at time t+steps_ahead
    df['future_has_issue'] = df.groupby('cluster_id')['has_issue'].shift(-steps_ahead)
    df['future_issue_type'] = df.groupby('cluster_id')['issue_type'].shift(-steps_ahead)
    
    # Define features to use - check actual column names in the dataset
    numerical_features = [
        'cluster_cpu_util_avg', 'cluster_cpu_util_max', 'cluster_cpu_util_min', 
        'cluster_cpu_util_std', 'node_count', 'sample_count',
        'node_count_change', 'cpu_change_rate', 'hour', 'day_of_week',
        'cpu_avg_1h', 'cpu_max_1h', 'cpu_std_1h', 'node_count_change_1h'
    ]
    
    # Update categorical feature names to match your dataset
    categorical_features = ['vmcategory_<lambda>', 'is_business_hours']
    
    # Drop rows with NaN in the target
    df = df.dropna(subset=['future_has_issue', 'future_issue_type'])
    
    # Split data temporally (not randomly)
    # Use earlier data for training and later data for testing
    split_time = df['timestamp'].quantile(0.8)
    
    train_data = df[df['timestamp'] <= split_time]
    test_data = df[df['timestamp'] > split_time]
    
    # Prepare X and y
    X_train = train_data[numerical_features + categorical_features]
    y_train_binary = train_data['future_has_issue']
    y_train_multiclass = train_data['future_issue_type']
    
    X_test = test_data[numerical_features + categorical_features]
    y_test_binary = test_data['future_has_issue']
    y_test_multiclass = test_data['future_issue_type']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Positive samples in training: {sum(y_train_binary)}")
    print(f"Positive samples in testing: {sum(y_test_binary)}")
    
    feature_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
    
    return (X_train, y_train_binary, y_train_multiclass, 
            X_test, y_test_binary, y_test_multiclass,
            feature_info)

def build_and_evaluate_models(X_train, y_train_binary, y_train_multiclass, 
                             X_test, y_test_binary, y_test_multiclass,
                             feature_info):
    """Build and evaluate machine learning models"""
    os.makedirs('models', exist_ok=True)
    
    # Create preprocessing pipeline
    numerical_features = feature_info['numerical_features']
    categorical_features = feature_info['categorical_features']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 1. Binary Classification Model (Issue vs No Issue)
    binary_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training binary classification model...")
    binary_clf.fit(X_train, y_train_binary)
    
    # Evaluate binary model
    binary_preds = binary_clf.predict(X_test)
    binary_proba = binary_clf.predict_proba(X_test)[:, 1]
    
    print("\nBinary Classification Performance:")
    print(classification_report(y_test_binary, binary_preds))
    
    # Plot binary classification confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test_binary, binary_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Binary Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('plots/binary_confusion_matrix.png')
    plt.close()
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test_binary, binary_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('plots/roc_curve.png')
    plt.close()
    
    # 2. Multiclass Classification Model (Issue Type)
    multiclass_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    print("\nTraining multiclass classification model...")
    multiclass_clf.fit(X_train, y_train_multiclass)
    
    # Evaluate multiclass model
    multiclass_preds = multiclass_clf.predict(X_test)
    
    print("\nMulticlass Classification Performance:")
    print(classification_report(y_test_multiclass, multiclass_preds))
    
    # Plot multiclass confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_multiclass, multiclass_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=multiclass_clf.classes_, 
               yticklabels=multiclass_clf.classes_)
    plt.title('Multiclass Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('plots/multiclass_confusion_matrix.png')
    plt.close()
    
    # Save the models
    print("\nSaving models...")
    joblib.dump(binary_clf, 'models/binary_classifier.pkl')
    joblib.dump(multiclass_clf, 'models/multiclass_classifier.pkl')
    joblib.dump(feature_info, 'models/feature_info.pkl')
    
    return binary_clf, multiclass_clf

def train_models():
    """Main function to train and evaluate models"""
    # Load data
    data = load_data()
    
    # Visualize data
    visualize_data(data)
    
    # Prepare features
    (X_train, y_train_binary, y_train_multiclass, 
     X_test, y_test_binary, y_test_multiclass,
     feature_info) = prepare_features(data)
    
    # Build and evaluate models
    binary_clf, multiclass_clf = build_and_evaluate_models(
        X_train, y_train_binary, y_train_multiclass,
        X_test, y_test_binary, y_test_multiclass,
        feature_info
    )
    
    print("Model training complete!")

if __name__ == "__main__":
    train_models()