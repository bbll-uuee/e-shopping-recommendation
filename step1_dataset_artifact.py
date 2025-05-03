from clearml import Task
import pandas as pd
import os
import zipfile

# Initialize ClearML task
task = Task.init(project_name='POC-ClearML', task_name='Step 1 - Load Data')

# Define parameters (can be overridden in ClearML UI)
params = {
    'zip_path': 'cleaned_amazon_data_final.csv.zip',  # ZIP file path
    'extract_dir': './extracted_data',                # Extract directory
    'csv_inside_zip': 'cleaned_amazon_data_final.csv' # CSV filename inside ZIP
}
task.connect(params)

# For remote execution - this line is key
task.execute_remotely()

# Ensure extract directory exists
os.makedirs(params['extract_dir'], exist_ok=True)

# Extract data file
with zipfile.ZipFile(params['zip_path'], 'r') as zip_ref:
    zip_ref.extractall(params['extract_dir'])

# Build complete CSV file path
csv_path = os.path.join(params['extract_dir'], params['csv_inside_zip'])

# Load CSV into DataFrame
df = pd.read_csv(csv_path)
print("✅ Data loaded successfully. Preview:")
print(df.head())

# Upload DataFrame as artifact for downstream tasks
task.upload_artifact(name='raw_data', artifact_object=df)

print("✅ Data uploaded as 'raw_data' artifact")
