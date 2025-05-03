from clearml import Task
import pandas as pd

# Initialize ClearML Task
task = Task.init(project_name='POC-ClearML', task_name='Step 1 - Load Data', task_type=Task.TaskTypes.data_processing)

# Read data
file_path = '/content/gdrive/My Drive/AIfans/cleaned_amazon_data_final.csv'
df = pd.read_csv(file_path)

# Preview sample
print("Loaded data:")
print(df.head())

# Upload raw DataFrame as artifact
task.upload_artifact(name='raw_data', artifact_object=df)
