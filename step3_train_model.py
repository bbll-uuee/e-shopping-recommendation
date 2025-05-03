from clearml import Task
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 3 - Train Clustering Model')

# Replace with your actual Step 2 task ID (visible in ClearML UI)
step2_task_id = ''

# Get the task from Step 2 and retrieve the processed data
source_task = Task.get_task(task_id=step2_task_id)
df_scaled = source_task.artifacts['processed_data'].get()

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(df_scaled)

# Visualize the clustering result
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue=cluster_labels, palette='Set2')
plt.title("KMeans Clustering Result")
plt.xlabel(df_scaled.columns[0])
plt.ylabel(df_scaled.columns[1])
plt.legend(title='Cluster')
plot_path = "kmeans_plot.png"
plt.savefig(plot_path)
plt.show()

# Upload the plot to ClearML
task.upload_artifact(name='kmeans_plot', artifact_object=plot_path)

print("✅ KMeans clustering completed. Plot uploaded as 'kmeans_plot'")
