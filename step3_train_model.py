from clearml import Task
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ✅ Init ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 3 - Train Clustering Model')

# ✅ Link to Step 2 by Task ID
step2_task_id = ''
source_task = Task.get_task(task_id=step2_task_id)

# ✅ Get processed data artifact
df_scaled = source_task.artifacts['processed_data'].get()

# ✅ Run KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(df_scaled)

# ✅ Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=labels, cmap='viridis')
plt.title("KMeans Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plot_path = 'kmeans_result.png'
plt.savefig(plot_path)

# ✅ Upload model and plot as artifacts
task.upload_artifact(name='kmeans_plot', artifact_object=plot_path)
task.upload_artifact(name='cluster_labels', artifact_object=labels.tolist())

print("✅ Clustering completed and results uploaded to ClearML.")
