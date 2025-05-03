from clearml import Task
import pandas as pd
import zipfile
import os

# Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 1 - Load Data', task_type=Task.TaskTypes.data_processing)

# Parameters (can be changed from UI or pipeline)
params = task.connect({
    'zip_path': '/content/gdrive/My Drive/AIfans/cleaned_amazon_data_final.zip',
    'csv_inside_zip': 'cleaned_amazon_data_final.csv',
    'extract_dir': '/content/extracted_data/'
})

# Ensure extraction directory exists
os.makedirs(params['extract_dir'], exist_ok=True)

# Unzip the file
with zipfile.ZipFile(params['zip_path'], 'r') as zip_ref:
    zip_ref.extractall(params['extract_dir'])

# Read extracted CSV file
csv_path = os.path.join(params['extract_dir'], params['csv_inside_zip'])
df = pd.read_csv(csv_path)

# Preview loaded data
print("Loaded data from extracted CSV:")
print(df.head())

# Upload entire DataFrame as artifact
task.upload_artifact(name='raw_data', artifact_object=df)
