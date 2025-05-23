import matplotlib.pyplot as plt
from clearml import Task, Logger, Dataset
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import time
import os
import logging
import shutil
import json
import seaborn as sns
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('assets', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Initialize the task
task = Task.init(
    project_name='examples',
    task_name='Final E-commerce Model Training',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

# Connect parameters - adapted from instructor's structure for e-commerce system
args = {
    'processed_dataset_id': 'cb8d133f414646ec8d1f755850230409',
    'hpo_task_id': '7b98e12fcab84fedbcadb378ac30d3c8',
    'test_queue': 'pipeline',
    'n_clusters_range_max': 5,
    'max_iter': 300,
    'outlier_threshold': 3.0,
    'min_interactions': 2,
    'clustering_algorithm': 'kmeans',
    'similarity_metric': 'cosine'
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute remotely
task.execute_remotely()

# Load dataset ID
dataset_id = task.get_parameter('General/processed_dataset_id') or args.get('processed_dataset_id') or "placeholder_dataset_id"
logger.info(f"Using dataset ID: {dataset_id}")

# Load HPO task and get best parameters
hpo_task_id = args.get('hpo_task_id')
hpo_task = Task.get_task(task_id=hpo_task_id)
logger.info(f"Retrieved HPO task: {hpo_task.name}")

try:
    best_params = hpo_task.get_parameter('best_parameters')
    best_silhouette_score = hpo_task.get_parameter('best_silhouette_score')

    if best_params is None:
        artifact_path = hpo_task.artifacts['best_parameters'].get_local_copy()
        with open(artifact_path, 'r') as f:
            best_results = json.load(f)
        best_params = best_results['best_experiment']['parameters']
        best_silhouette_score = best_results['best_experiment'].get('silhouette_score')

    # Update args
    for param in ['n_clusters_range_max', 'max_iter', 'outlier_threshold', 'min_interactions', 'clustering_algorithm', 'similarity_metric']:
        args[param] = best_params.get(param, args[param])

except Exception as e:
    logger.warning(f"Failed to get best HPO parameters: {e}")
    best_silhouette_score = None

# Load data
try:
    preprocessing_task = Task.get_task(task_id=dataset_id)
    df = pd.read_csv(preprocessing_task.artifacts['processed_full_data'].get_local_copy())
    clustering_df = pd.read_csv(preprocessing_task.artifacts['clustering_features'].get_local_copy())
    scaler = joblib.load(preprocessing_task.artifacts['standard_scaler'].get_local_copy())
    clustering_features_scaled = scaler.transform(clustering_df)
except Exception as e:
    logger.error(f"Failed to load dataset from task {dataset_id}: {e}")
    raise

# Initialize model
model_cls = KMeans if args['clustering_algorithm'] == 'kmeans' else MiniBatchKMeans
model = model_cls(n_clusters=args['n_clusters_range_max'], max_iter=args['max_iter'], n_init=10, random_state=42)

# Train model
start_time = time.time()
for iteration in tqdm(range(min(args['max_iter'], 100)), desc="Training Progress"):
    if args['clustering_algorithm'] == 'minibatch_kmeans':
        batch_indices = np.random.choice(len(clustering_features_scaled), min(1000, len(clustering_features_scaled)), replace=False)
        model.partial_fit(clustering_features_scaled[batch_indices])
    if iteration % 10 == 0:
        task.get_logger().report_scalar('training', 'progress', value=(iteration + 1) * 100 / args['max_iter'], iteration=iteration)

cluster_labels = model.fit_predict(clustering_features_scaled)
training_time = time.time() - start_time
df['Cluster'] = cluster_labels

# Evaluate metrics
silhouette = silhouette_score(clustering_features_scaled, cluster_labels)
calinski = calinski_harabasz_score(clustering_features_scaled, cluster_labels)
davies_bouldin = davies_bouldin_score(clustering_features_scaled, cluster_labels)
inertia = model.inertia_

# Log metrics
logger_metrics = task.get_logger()
logger_metrics.report_scalar('validation', 'silhouette_score', silhouette, iteration=0)
logger_metrics.report_scalar('validation', 'calinski_harabasz_score', calinski, iteration=0)
logger_metrics.report_scalar('validation', 'davies_bouldin_score', davies_bouldin, iteration=0)
logger_metrics.report_scalar('validation', 'inertia', inertia, iteration=0)

# Cluster distribution plot
plt.figure(figsize=(10, 6))
cluster_counts = df['Cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, alpha=0.7, color='skyblue', edgecolor='black')
plt.title(f'Cluster Distribution (Silhouette Score: {silhouette:.3f})')
plt.xlabel('Cluster')
plt.ylabel('Customers')
for i, count in enumerate(cluster_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom')
logger_metrics.report_matplotlib_figure('Cluster Distribution', 'cluster_distribution', plt.gcf(), 0)
plt.close()

# Feature histograms
plt.figure(figsize=(15, 10))
features = ['discounted_price', 'actual_price', 'rating', 'Profit']
for i, feat in enumerate(features[:4]):
    plt.subplot(2, 2, i + 1)
    for cluster in range(args['n_clusters_range_max']):
        cluster_data = df[df['Cluster'] == cluster][feat]
        plt.hist(cluster_data, bins=20, alpha=0.6, label=f'Cluster {cluster}')
    plt.title(f'{feat.title()} by Cluster')
    plt.legend()
logger_metrics.report_matplotlib_figure('Feature Analysis', 'feature_distributions', plt.gcf(), 0)
plt.close()

# Build recommendation matrix
try:
    user_interactions = df.groupby('user_id').size()
    product_interactions = df.groupby('product_name').size()
    valid_users = user_interactions[user_interactions >= args['min_interactions']].index
    valid_products = product_interactions[product_interactions >= args['min_interactions']].index
    recommendation_df = df[(df['user_id'].isin(valid_users)) & (df['product_name'].isin(valid_products))]

    user_product_matrix = recommendation_df.pivot_table(index='user_id', columns='product_name', values='Quantity', aggfunc='sum').fillna(0)
    if args['similarity_metric'] == 'cosine':
        product_similarity = cosine_similarity(user_product_matrix.T)
    else:
        from sklearn.metrics.pairwise import euclidean_distances
        product_similarity = 1 / (1 + euclidean_distances(user_product_matrix.T))

    product_similarity_df = pd.DataFrame(product_similarity, index=user_product_matrix.columns, columns=user_product_matrix.columns)

    logger_metrics.report_scalar('recommendation', 'total_interactions', len(recommendation_df), iteration=0)
    logger_metrics.report_scalar('recommendation', 'unique_users', len(valid_users), iteration=0)
    logger_metrics.report_scalar('recommendation', 'unique_products', len(valid_products), iteration=0)
    sparsity = (user_product_matrix == 0).sum().sum() / (user_product_matrix.size)
    logger_metrics.report_scalar('recommendation', 'matrix_sparsity', sparsity, iteration=0)

except Exception as e:
    logger.warning(f"Failed to build recommendation system: {e}")
    user_product_matrix = None
    product_similarity_df = None

# Save models
joblib.dump(model, 'final_clustering_model.pkl')
joblib.dump(scaler, 'final_scaler.pkl')
df.to_csv('final_clustered_data.csv', index=False)
task.upload_artifact('clustering_model', 'final_clustering_model.pkl')
task.upload_artifact('scaler', 'final_scaler.pkl')
task.upload_artifact('clustered_data', 'final_clustered_data.csv')

if user_product_matrix is not None:
    joblib.dump(user_product_matrix, 'user_product_matrix.pkl')
    joblib.dump(product_similarity_df, 'product_similarity_matrix.pkl')
    task.upload_artifact('user_product_matrix', 'user_product_matrix.pkl')
    task.upload_artifact('product_similarity_matrix', 'product_similarity_matrix.pkl')

# Save summary
summary = {
    'model_type': 'E-commerce Clustering + Recommendation System',
    'n_clusters': args['n_clusters_range_max'],
    'performance_metrics': {
        'silhouette_score': silhouette,
        'calinski_harabasz_score': calinski,
        'davies_bouldin_score': davies_bouldin,
        'inertia': inertia
    },
    'training_time_seconds': training_time,
    'samples_processed': len(df),
    'best_hpo_score': best_silhouette_score
}
with open('final_model_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)
task.upload_artifact('model_summary', 'final_model_summary.json')

logger.info("Model training complete. Artifacts saved and uploaded.")
