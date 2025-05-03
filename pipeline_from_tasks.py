from clearml import PipelineDecorator, Task

@PipelineDecorator.pipeline(
    name='POC Amazon Data Clustering Pipeline',
    project='POC-ClearML',
    version='1.0',
)
def run_pipeline():
    # Step 1: Load Data
    step1 = PipelineDecorator.task(
        name='Step 1 - Load Data',
        project='POC-ClearML',
        task_name='Step 1 - Load Data',
        script='step1_load_data.py',
        task_type=Task.TaskTypes.data_processing,
    )

    # Step 2: Data Preprocessing
    step2 = PipelineDecorator.task(
        name='Step 2 - Data Preprocessing',
        project='POC-ClearML',
        task_name='Step 2 - Data Preprocessing',
        script='step2_preprocessing.py',
        task_type=Task.TaskTypes.data_processing,
        parent=step1
    )

    # Step 3: Clustering Mode
    step3 = PipelineDecorator.task(
        name='Step 3 - Clustering Model',
        project='POC-ClearML',
        task_name='Step 3 - Train Clustering Model',
        script='step3_clustering_model.py',
        task_type=Task.TaskTypes.training,
        parent=step2
    )

if __name__ == '__main__':
    run_pipeline()
