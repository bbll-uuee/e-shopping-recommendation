# /content/e-shopping-recommendation/cleaned_amazon_data_final.csv.zip

from clearml import Task, Dataset

# ✅ Init ClearML Task
task = Task.init(
    project_name='POC-ClearML',
    task_name='Step 1 - Upload Dataset',
    reuse_last_task_id=False
)

# ✅ Dataset version control (for ClearML Datasets tab)
zip_file_path = '/content/e-shopping-recommendation/cleaned_amazon_data_final.csv.zip'
dataset = Dataset.create(
    dataset_name='Amazon Dataset - Final ZIP',
    dataset_project='POC-ClearML'
)
dataset.add_files(path=zip_file_path)
dataset.upload()
dataset.finalize()

# ✅ Also upload the same file as a Task Artifact
task.upload_artifact(name='raw_dataset_zip', artifact_object=zip_file_path)

# ✅ Print for reference
print(f"✅ Dataset uploaded to ClearML Dataset with ID: {dataset.id}")
print(f"✅ ZIP also uploaded as artifact: 'raw_dataset_zip'")
print(f"📌 Step 1 Task ID (for Step 2 linking): {task.id}")
