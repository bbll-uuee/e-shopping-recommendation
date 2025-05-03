from clearml import Task
import zipfile, os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ✅ Start a new ClearML task
task = Task.init(
    project_name='POC-ClearML',
    task_name='Step 2 - Data Preprocessing',
    reuse_last_task_id=False
)

# ✅ Step 1 task ID: manually insert the task ID from Step 1
step1_task_id = ''
source_task = Task.get_task(task_id=step1_task_id)

# ✅ Download and extract ZIP artifact
zip_path = source_task.artifacts['raw_dataset_zip'].get_local_copy()
extract_dir = './extracted_data'
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

# ✅ Read CSV from extracted files
csv_file = os.path.join(extract_dir, 'cleaned_amazon_data_final.csv')  # replace if needed
df = pd.read_csv(csv_file)
df.fillna(0, inplace=True)

# ✅ Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le.classes_

# ✅ Normalize numeric values
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

# ✅ Upload processed DataFrame as artifact
task.upload_artifact(name='processed_data', artifact_object=df_scaled)

print("✅ Step 2 complete: Preprocessed data uploaded as 'processed_data'")
print(f"📌 Step 2 Task ID: {task.id}")
