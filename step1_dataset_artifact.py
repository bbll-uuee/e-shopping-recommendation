# /content/e-shopping-recommendation/cleaned_amazon_data_final.csv.zip

from clearml import Task, Dataset

# ✅ Init Task
task = Task.init(
    project_name='POC-ClearML',
    task_name='Step 1 - Upload Dataset',
    reuse_last_task_id=False
)

# ✅ Upload Dataset to ClearML Dataset (for version control)
dataset = Dataset.create(
    dataset_name='Amazon Dataset - Final ZIP',
    dataset_project='POC-ClearML'
)
dataset.add_files(path='/content/e-shopping-recommendation/cleaned_amazon_data_final.csv.zip')  # ZIP file path
dataset.upload()
dataset.finalize()

# ✅ Also upload ZIP as artifact for downstream use
zip_path = 'cleaned_amazon_data_final.zip'
task.upload_artifact(name='raw_dataset_zip', artifact_object=zip_path)

# ✅ Print both IDs
print(f"✅ Dataset uploaded to ClearML Dataset with ID: {dataset.id}")
print(f"✅ ZIP also uploaded as task artifact: 'raw_dataset_zip'")
print(f"📌 Step 1 Task ID (for linking): {task.id}")
