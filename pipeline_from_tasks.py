from clearml import PipelineDecorator, Task

@PipelineDecorator.pipeline(
    name='POC Manual Artifact Pipeline',
    project='POC-ClearML',
    version='1.0',
    default_execution_queue='pipline'  # match your agent queue name
)
def run_pipeline():
    # Step 1: Run data loading task
    step1 = PipelineDecorator.task(
        name='Step 1 - Load Data',
        project='POC-ClearML',
        task_name='Step 1 - Load Data',
        script='step1_dataset_artifact.py',
        task_type=Task.TaskTypes.data_processing,
        queue='pipline'
    )

    # Step 2: Preprocessing (uses manually set Step 1 task ID)
    step2 = PipelineDecorator.task(
        name='Step 2 - Data Preprocessing',
        project='POC-ClearML',
        task_name='Step 2 - Data Preprocessing',
        script='step2_data_preprocessing.py',
        task_type=Task.TaskTypes.data_processing,
        queue='pipline'
    )

    # Step 3: Clustering (uses manually set Step 2 task ID)
    step3 = PipelineDecorator.task(
        name='Step 3 - Clustering Model',
        project='POC-ClearML',
        task_name='Step 3 - Train Clustering Model',
        script='step3_train_model.py',
        task_type=Task.TaskTypes.training,
        queue='pipline'
    )

if __name__ == '__main__':
    run_pipeline()
