from clearml import Task
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 初始化ClearML任务
task = Task.init(project_name='POC-ClearML', task_name='Step 2 - Data Preprocessing')

# 定义参数
args = {
    'dataset_task_id': '',  # 将在pipeline中填充
    'test_size': 0.2,
    'random_state': 42
}
task.connect(args)

# 用于远程执行 - 这行是关键
task.execute_remotely()

# 获取上一阶段的数据
print(f"获取任务ID为{args['dataset_task_id']}的数据")
source_task = Task.get_task(task_id=args['dataset_task_id'])
df = source_task.artifacts['raw_data'].get()

# 填充缺失值
df.fillna(0, inplace=True)

# 编码分类特征
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# 标准化数值特征
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
if not num_cols.empty:
    df[num_cols] = scaler.fit_transform(df[num_cols])

# 上传处理后的数据
task.upload_artifact(name='processed_data', artifact_object=df)
print("✅ 预处理完成。预览:")
print(df.head())
