from clearml import Task
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 初始化ClearML任务
task = Task.init(project_name='POC-ClearML', task_name='Step 3 - Train Clustering Model')

# 定义参数
args = {
    'dataset_task_id': '',  # 将在pipeline中填充
    'n_clusters': 3,
    'random_state': 42
}
task.connect(args)

# 用于远程执行 - 这行是关键
task.execute_remotely()

# 获取处理过的数据
print(f"获取任务ID为{args['dataset_task_id']}的处理后数据")
source_task = Task.get_task(task_id=args['dataset_task_id'])
df_scaled = source_task.artifacts['processed_data'].get()

# 运行KMeans聚类
kmeans = KMeans(n_clusters=args['n_clusters'], random_state=args['random_state'])
labels = kmeans.fit_predict(df_scaled)

# 可视化聚类
plt.figure(figsize=(10, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=labels, cmap='viridis')
plt.title("KMeans Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plot_path = 'kmeans_result.png'
plt.savefig(plot_path)

# 上传模型和图表作为工件
task.upload_artifact(name='kmeans_plot', artifact_object=plot_path)
task.upload_artifact(name='cluster_labels', artifact_object=labels.tolist())

print("✅ 聚类完成，结果已上传到ClearML。")
