from clearml import Task
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 2 - Data Preprocessing')

# Replace this with your actual Step 1 task ID (check in ClearML UI)
step1_task_id = ''

# Connect to Step 1 task and retrieve the artifact
source_task = Task.get_task(task_id=step1_task_id)
df = source_task.artifacts['raw_data'].get()  # Returns a DataFrame

# Fill missing values (basic handling)
df.fillna(0, inplace=True)

# Label encode all categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le.classes_

# Standardize all numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

# Upload preprocessed data as an artifact
task.upload_artifact(name='processed_data', artifact_object=df_scaled)

print("✅ Data preprocessing completed and uploaded as 'processed_data'")
