import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

from clearml import Task
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ClearML Task Initialization === #
task = Task.init(
    project_name='ecommerce_recommendation',
    task_name='Final Model Training',
    task_type=Task.TaskTypes.training
)

task.execute_remotely(queue_name='pipeline')  # Optional

# === Parameters === #
args = {
    'processed_dataset_id': '8f38e287110f45e8ba0733ed1ce04471',  
    'hpo_task_id': 'ebc118e6b3ad410c9132f3153a649393',          
    'recommendation_top_n': 5
}
args = task.connect(args)
logger.info(f"Parameters: {args}")

# === Load dataset from ClearML === #
dataset = Task.get_task(task_id=args['processed_dataset_id'])
dataset_path = dataset.artifacts['processed_full_data'].get_local_copy()
df = pd.read_csv(dataset_path)
logger.info(f"Loaded dataset with shape: {df.shape}")

# === Load best hyperparameters from HPO task === #
hpo_task = Task.get_task(task_id=args['hpo_task_id'])
best_params = hpo_task.get_parameters()

n_clusters = int(best_params.get('General/n_clusters', 5))
max_iter = int(best_params.get('General/max_iter', 300))
similarity_metric = best_params.get('General/similarity_metric', 'cosine')
min_interactions = int(best_params.get('General/min_interactions', 1))
logger.info(f"Using best hyperparameters: K={n_clusters}, MaxIter={max_iter}, Sim={similarity_metric}")

# === Clustering === #
features = ['discounted_price', 'actual_price', 'rating', 'rating_count', 'Profit']
X = df[features]

kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
silhouette = silhouette_score(X, df['Cluster'])
logger.info(f"Final silhouette score: {silhouette:.4f}")

# === Recommendation Matrix === #
interactions = df.groupby(['user_id', 'product_name'])['Quantity'].sum().reset_index()
matrix = interactions.pivot(index='user_id', columns='product_name', values='Quantity').fillna(0)

if similarity_metric == 'cosine':
    similarity = cosine_similarity(matrix)
else:
    from sklearn.metrics.pairwise import euclidean_distances
    similarity = 1 / (1 + euclidean_distances(matrix))

user_similarity_df = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)
logger.info("Computed user-user similarity matrix")

# === Save Artifacts === #
model_path = 'final_kmeans_model.pkl'
data_path = 'data_with_clusters.csv'
similarity_path = 'user_similarity_matrix.pkl'

joblib.dump(kmeans, model_path)
df.to_csv(data_path, index=False)
joblib.dump(user_similarity_df, similarity_path)

logger.info("Saved all final artifacts")

task.upload_artifact('final_kmeans_model', model_path)
task.upload_artifact('clustered_data', data_path)
task.upload_artifact('user_similarity_matrix', similarity_path)
task.upload_artifact('final_silhouette_score', silhouette)
task.upload_artifact('final_params', {
    'n_clusters': n_clusters,
    'max_iter': max_iter,
    'similarity_metric': similarity_metric,
    'silhouette_score': silhouette,
    'timestamp': datetime.now().isoformat()
})

logger.info("Final model training completed successfully.")
