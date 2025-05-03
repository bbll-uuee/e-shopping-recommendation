from clearml import Task
import pandas as pd
import os
import zipfile

# Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 1 - Load Data')

# Define parameters (can be overridden in the ClearML UI)
params = {
    'zip_path': './content/e-shopping-recommendation/cleaned_amazon_data_final.csv.zip',           # Path to the .zip file
    'extract_dir': './content/e-shopping-recommendation/extracted_data',                       # Directory to extract contents
    'csv_inside_zip': 'cleaned_amazon_data_final.csv'        # CSV file name inside the ZIP
}
task.connect(params)

# Ensure extraction directory exists
os.makedirs(params['extract_dir'], exist_ok=True)

# Unzip the data file
with zipfile.ZipFile(params['zip_path'], 'r') as zip_ref:
    zip_ref.extractall(params['extract_dir'])

# Construct full CSV file path
csv_path = os.path.join(params['extract_dir'], params['csv_inside_zip'])

# Load CSV into DataFrame
df = pd.read_csv(csv_path)
print("✅ Data loaded successfully. Preview:")
print(df.head())

# Upload the DataFrame as an artifact for downstream tasks
task.upload_artifact(name='raw_data', artifact_object=df)

print("✅ Data uploaded to ClearML as artifact: 'raw_data'")
