from clearml import Task
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Start a ClearML task
task = Task.init(
    project_name='POC-ClearML',
    task_name='Step 3 - Train KMeans Model',
    reuse_last_task_id=False
)

# ✅ Step 2 task ID: insert Step 2's task ID
step2_task_id = ''
source_task = Task.get_task(task_id=step2_task_id)

# ✅ Download DataFrame
df_scaled = source_task.artifacts['processed_data'].get()

# ✅ Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(df_scaled)

# ✅ Visualize results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_scaled.iloc[:, 0], y=df_scaled.iloc[:, 1], hue=labels, palette='Set2')
plt.title('KMeans Clustering Result')
plt.xlabel(df_scaled.columns[0])
plt.ylabel(df_scaled.columns[1])
plt.legend(title='Cluster')
plot_path = 'kmeans_plot.png'
plt.savefig(plot_path)
plt.show()

# ✅ Upload clustering plot
task.upload_artifact(name='kmeans_plot', artifact_object=plot_path)

print("✅ Step 3 complete: KMeans model trained and plot uploaded.")
