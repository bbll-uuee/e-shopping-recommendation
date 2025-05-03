from clearml import Task
import pandas as pd
import os
import zipfile

# 初始化ClearML任务
task = Task.init(project_name='POC-ClearML', task_name='Step 1 - Load Data')

# 定义参数(可在ClearML UI中覆盖)
params = {
    'zip_path': 'cleaned_amazon_data_final.csv.zip',  # ZIP文件路径
    'extract_dir': './extracted_data',                # 解压目录
    'csv_inside_zip': 'cleaned_amazon_data_final.csv' # ZIP内CSV文件名
}
task.connect(params)

# 用于远程执行 - 这行是关键
task.execute_remotely()

# 确保解压目录存在
os.makedirs(params['extract_dir'], exist_ok=True)

# 解压数据文件
with zipfile.ZipFile(params['zip_path'], 'r') as zip_ref:
    zip_ref.extractall(params['extract_dir'])

# 构建完整CSV文件路径
csv_path = os.path.join(params['extract_dir'], params['csv_inside_zip'])

# 将CSV加载到DataFrame
df = pd.read_csv(csv_path)
print("✅ 数据加载成功。预览:")
print(df.head())

# 将DataFrame上传为下游任务的工件
task.upload_artifact(name='raw_data', artifact_object=df)

print("✅ 数据已上传为'raw_data'工件")
