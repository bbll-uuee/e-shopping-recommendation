from clearml import Task
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ✅ Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 2 - Data Preprocessing')

# ✅ Link Step 1 task by its ID
step1_task_id = ''
source_task = Task.get_task(task_id=step1_task_id)

# ✅ Get DataFrame artifact directly
df = source_task.artifacts['raw_data'].get()

# ✅ Fill missing values
df.fillna(0, inplace=True)

# ✅ Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le.classes_

# ✅ Standardize numerical features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

# ✅ Upload processed data
task.upload_artifact(name='processed_data', artifact_object=df_scaled)

print("✅ Step 2 completed. Preprocessed data uploaded.")
