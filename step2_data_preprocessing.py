from clearml import Task
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize ClearML task
task = Task.init(project_name='POC-ClearML', task_name='Step 2 - Data Preprocessing')

# Define parameters
args = {
    'dataset_task_id': '3765d06737ac4834ae0192e42feead69',  # Will be populated in pipeline
    'test_size': 0.2,
    'random_state': 42
}
task.connect(args)

# For remote execution - this line is key
task.execute_remotely()

# Get data from previous stage
print(f"Retrieving data from task ID {args['dataset_task_id']}")
source_task = Task.get_task(task_id=args['dataset_task_id'])
df = source_task.artifacts['raw_data'].get()

# Fill missing values
df.fillna(0, inplace=True)

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
if not num_cols.empty:
    df[num_cols] = scaler.fit_transform(df[num_cols])

# Upload processed data
task.upload_artifact(name='processed_data', artifact_object=df)
print("✅ Preprocessing completed. Preview:")
print(df.head())
