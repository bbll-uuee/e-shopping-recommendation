from clearml import Task
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# This is Step 2 Task
task = Task.init(project_name='POC-ClearML', task_name='Step 2 - Data Preprocessing', task_type=Task.TaskTypes.data_processing)

# ✅ Manually fetch artifact from Step 1 by ID
source_task_id = ''  # Step 1 Task ID
source_task = Task.get_task(task_id=source_task_id)

# ✅ Get artifact as DataFrame directly
df = source_task.artifacts['raw_data'].get()

# 🧹 Fill missing values
df.fillna(0, inplace=True)

# 🔠 Label encoding
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le.classes_

# 🔢 Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 📄 Convert back to DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

# 📤 Upload artifact
task.upload_artifact(name='processed_data', artifact_object=df_scaled)
