from clearml import Task
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Initialize ClearML task
task = Task.init(project_name='POC-ClearML', task_name='Step 3 - Train Clustering Model')

# Define parameters
args = {
    'dataset_task_id': '',  # Will be populated in pipeline
    'n_clusters': 3,
    'random_state': 42
}
task.connect(args)

# For remote execution - this line is key
task.execute_remotely()

# Get processed data
print(f"Retrieving processed data from task ID {args['dataset_task_id']}")
source_task = Task.get_task(task_id=args['dataset_task_id'])
df_scaled = source_task.artifacts['processed_data'].get()

# Run KMeans clustering
kmeans = KMeans(n_clusters=args['n_clusters'], random_state=args['random_state'])
labels = kmeans.fit_predict(df_scaled)

# Visualize clustering
plt.figure(figsize=(10, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=labels, cmap='viridis')
plt.title("KMeans Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plot_path = 'kmeans_result.png'
plt.savefig(plot_path)

# Upload model and chart as artifacts
task.upload_artifact(name='kmeans_plot', artifact_object=plot_path)
task.upload_artifact(name='cluster_labels', artifact_object=labels.tolist())

print("✅ Clustering completed, results uploaded to ClearML.")
