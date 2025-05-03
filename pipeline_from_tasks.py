from clearml import PipelineDecorator, Task

@PipelineDecorator.pipeline(
    name='POC Amazon Data Clustering Pipeline',
    project='POC-ClearML',
    version='1.0',
    default_execution_queue='pipline'  # ✅ make sure this matches your clearml-agent queue
)
def run_pipeline():
    # Step 1: Load and upload raw CSV data
    step1 = PipelineDecorator.task(
        name='Step 1 - Load Data',
        project='POC-ClearML',
        task_name='Step 1 - Load Data',
        script='step1_dataset_artifact.py',
        task_type=Task.TaskTypes.data_processing,
        queue='pipline'
    )

    # Step 2: Preprocess Data
    step2 = PipelineDecorator.task(
        name='Step 2 - Preprocessing',
        project='POC-ClearML',
        task_name='Step 2 - Data Preprocessing',
        script='step2_data_preprocessing.py',
        task_type=Task.TaskTypes.data_processing,
        parent=step1,
        queue='pipline'
    )

    # Step 3: Train Clustering Model
    step3 = PipelineDecorator.task(
        name='Step 3 - Train Clustering Model',
        project='POC-ClearML',
        task_name='Step 3 - Train Clustering Model',
        script='step3_train_model.py',
        task_type=Task.TaskTypes.training,
        parent=step2,
        queue='pipline'
    )

if __name__ == '__main__':
    run_pipeline()
