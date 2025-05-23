from clearml import PipelineController

def run_pipeline():
    pipe = PipelineController(
        name="Ecommerce Recommendation Pipeline",
        project="ecommerce_recommendation",
        version="1.0.0",
        add_pipeline_tags=True
    )

    # Default execution queue
    pipe.set_default_execution_queue("pipeline")

    # Step 1: Create dataset
    pipe.add_step(
        name="step1_create_dataset",
        base_task_project="examples",
        base_task_name="Pipeline step 1 dataset artifact",
        parameter_override={}
    )

    # Step 2: Data preprocessing
    pipe.add_step(
        name="step2_preprocess",
        base_task_project="examples",
        base_task_name="Pipeline step 2 process dataset",
        parents=["step1_create_dataset"],
        parameter_override={
            "General/dataset_task_id": "${step1_create_dataset.id}"
        }
    )

    # Step 3: Initial model training
    pipe.add_step(
        name="step3_initial_train",
        base_task_project="examples",
        base_task_name="Pipeline step 3 train model",
        parents=["step2_preprocess"],
        parameter_override={
            "General/processed_dataset_id": "${step2_preprocess.parameters.General/processed_dataset_id}"
        }
    )

    # Step 4: Hyperparameter optimization (HPO)
    pipe.add_step(
        name="step4_hpo",
        base_task_project="ecommerce_recommendation",
        base_task_name="HPO: Cluster & Recommend",
        parents=["step3_initial_train"],
        parameter_override={
            "General/processed_dataset_id": "${step2_preprocess.parameters.General/processed_dataset_id}"
        }
    )

    # Step 5: Final model training (using best parameters)
    pipe.add_step(
        name="step5_final_model",
        base_task_project="ecommerce_recommendation",
        base_task_name="Final Model Training",
        parents=["step4_hpo"],
        parameter_override={
            "General/processed_dataset_id": "${step2_preprocess.parameters.General/processed_dataset_id}",
            "General/hpo_task_id": "${step4_hpo.id}"
        }
    )

    # Start the pipeline
    pipe.start(queue="pipeline")

if __name__ == "__main__":
    run_pipeline()
