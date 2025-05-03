from clearml import Task
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 2 - Data Preprocessing', task_type=Task.TaskTypes.data_processing)

# Retrieve artifact from previous step
raw_data = task.artifacts['raw_data'].get()
df = pd.read_csv(raw_data)

# Fill missing values
df.fillna(0, inplace=True)

# Label encoding for categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le.classes_

# Standardize numeric features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Convert to DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

# Upload processed data
task.upload_artifact(name='processed_data', artifact_object=df_scaled)
