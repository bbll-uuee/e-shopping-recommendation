from clearml import Task
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 3 - Train Clustering Model', task_type=Task.TaskTypes.training)

# ✅ Get artifact from previous step using known Task ID
source_task_id = ''  # Replace with your actual Step 2 task ID
source_task = Task.get_task(task_id=source_task_id)

# ✅ Directly get DataFrame artifact
df_scaled = source_task.artifacts['processed_data'].get()

# 🧠 Train KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(df_scaled)

# 📊 Plot clustering result
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue=labels, palette='viridis')
plt.title('KMeans Clustering Result')
plt.savefig('kmeans_plot.png')

# 📤 Upload plot as artifact
task.upload_artifact(name='kmeans_plot', artifact_object='kmeans_plot.png')
