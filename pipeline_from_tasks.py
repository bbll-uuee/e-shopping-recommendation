from clearml import Task
from clearml.automation import PipelineController
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue configuration
EXECUTION_QUEUE = "pipeline"

def run_pipeline():
    """
    Complete 5-Step E-commerce Recommendation System Pipeline
    Step 1: Dataset Creation
    Step 2: Data Preprocessing  
    Step 3: Model Training
    Step 4: Hyperparameter Optimization (HPO)
    Step 5: Final Model Training
    """
    logger.info("🚀 Starting Complete E-commerce Recommendation System Pipeline...")
    logger.info("5-Step Pipeline: Data → Process → Train → HPO → Final Model")
    
    # Initialize pipeline controller
    pipe = PipelineController(
        name="E-commerce_Complete_Pipeline", 
        project="examples",
        version="1.0.0", 
        add_pipeline_tags=False
    )
    
    # Set execution queue
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    logger.info(f"Set default execution queue to: {EXECUTION_QUEUE}")
    
    # Step 1: Dataset Creation
    logger.info("Adding Step 1: Dataset Creation")
    pipe.add_step(
        name="step1_dataset",
        base_task_project="examples",
        base_task_name="Pipeline step 1 dataset artifact",
        execution_queue=EXECUTION_QUEUE
    )
    
    # Step 2: Data Preprocessing  
    logger.info("Adding Step 2: Data Preprocessing")
    pipe.add_step(
        name="step2_preprocessing",
        parents=["step1_dataset"],
        base_task_project="examples", 
        base_task_name="Pipeline step 2 process dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${step1_dataset.id}",
            "General/test_size": 0.25,
            "General/random_state": 42,
            "General/remove_outliers": True,
            "General/outlier_threshold": 3.0,
            "General/feature_scaling": True,
            "General/handle_missing_values": True
        }
    )
    
    # Step 3: Model Training
    logger.info("Adding Step 3: Model Training")
    pipe.add_step(
        name="step3_training", 
        parents=["step2_preprocessing"],
        base_task_project="examples",
        base_task_name="Pipeline step 3 train model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${step2_preprocessing.id}",
            "General/clustering_algorithm": "kmeans",
            "General/n_clusters_range": [2, 10],
            "General/max_iter": 300,
            "General/n_init": 10,
            "General/optimize_clusters": True,
            "General/train_recommendation": True,
            "General/similarity_metric": "cosine",
            "General/min_interactions": 1
        }
    )
    
    # Step 4: Hyperparameter Optimization (HPO)
    logger.info("Adding Step 4: Hyperparameter Optimization")
    pipe.add_step(
        name="step4_hpo",
        parents=["step3_training", "step2_preprocessing"],
        base_task_project="examples",
        base_task_name="HPO: E-commerce Recommendation System",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/base_train_task_id": "${step3_training.id}",
            "General/processed_dataset_id": "${step2_preprocessing.id}",
            "General/num_trials": 5,
            "General/time_limit_minutes": 25,
            "General/run_as_service": False,
            "General/test_queue": EXECUTION_QUEUE,
            # HPO参数
            "General/n_clusters_max": 10,
            "General/max_iter": 300,
            "General/outlier_threshold": 3.0,
            "General/min_interactions": 2,
            "General/clustering_algorithm": "kmeans",
            "General/similarity_metric": "cosine"
        }
    )
    
    # Step 5: Final Model Training
    logger.info("Adding Step 5: Final Model Training")
    pipe.add_step(
        name="step5_final_model",
        parents=["step4_hpo", "step2_preprocessing"],
        base_task_project="examples",
        base_task_name="Final E-commerce Model Training",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${step2_preprocessing.id}",
            "General/hpo_task_id": "${step4_hpo.id}",
            "General/test_queue": EXECUTION_QUEUE,
            # 最终模型参数
            "General/n_clusters_range_max": 5,
            "General/max_iter": 300,
            "General/outlier_threshold": 3.0,
            "General/min_interactions": 2,
            "General/clustering_algorithm": "kmeans",
            "General/similarity_metric": "cosine"
        }
    )
    
    # Pipeline summary
    logger.info("="*60)
    logger.info("COMPLETE E-COMMERCE PIPELINE STRUCTURE")
    logger.info("="*60)
    logger.info("📊 Step 1: Dataset Creation")
    logger.info("   └─ Creates Amazon e-commerce sample dataset")
    logger.info("🔧 Step 2: Data Preprocessing")
    logger.info("   └─ Cleans data, handles outliers, feature engineering")
    logger.info("🤖 Step 3: Model Training") 
    logger.info("   └─ Trains clustering + recommendation models")
    logger.info("🎯 Step 4: Hyperparameter Optimization")
    logger.info("   └─ Finds best parameters using HPO")
    logger.info("🏆 Step 5: Final Model Training")
    logger.info("   └─ Trains final model with optimized parameters")
    logger.info("="*60)
    logger.info(f"Execution Queue: {EXECUTION_QUEUE}")
    logger.info("="*60)
    
    # Start the pipeline
    logger.info("🚀 Starting complete 5-step pipeline execution...")
    logger.info("This will take approximately 30-45 minutes to complete")
    
    pipe.start_locally()
    
    logger.info("✅ Complete pipeline started successfully!")
    logger.info("📋 Monitor progress in ClearML UI")
    logger.info("🔄 Pipeline Flow:")
    logger.info("   step1_dataset → step2_preprocessing → step3_training")
    logger.info("                                    ↓")
    logger.info("   step5_final_model ← step4_hpo ←──┘")


def run_simple_pipeline():
    """
    Simplified 3-step pipeline for testing
    """
    logger.info("🔧 Starting SIMPLIFIED 3-step pipeline for testing...")
    
    pipe = PipelineController(
        name="E-commerce_Simple_Test", 
        project="examples",
        version="1.0.0-simple",
        add_pipeline_tags=False
    )
    
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    
    # Only first 3 steps
    pipe.add_step(
        name="test_step1",
        base_task_project="examples",
        base_task_name="Pipeline step 1 dataset artifact",
        execution_queue=EXECUTION_QUEUE
    )
    
    pipe.add_step(
        name="test_step2",
        parents=["test_step1"],
        base_task_project="examples",
        base_task_name="Pipeline step 2 process dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${test_step1.id}",
            "General/test_size": 0.2
        }
    )
    
    pipe.add_step(
        name="test_step3",
        parents=["test_step2"],
        base_task_project="examples",
        base_task_name="Pipeline step 3 train model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${test_step2.id}",
            "General/optimize_clusters": True,
            "General/train_recommendation": True
        }
    )
    
    logger.info("Starting simplified 3-step pipeline...")
    pipe.start_locally()
    logger.info("✅ Simplified pipeline started")


def run_hpo_test():
    """
    Test HPO step separately
    """
    logger.info("🎯 Testing HPO step separately...")
    
    pipe = PipelineController(
        name="E-commerce_HPO_Test", 
        project="examples",
        version="1.0.0-hpo-test",
        add_pipeline_tags=False
    )
    
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    
    # Just the HPO step with a base task
    pipe.add_step(
        name="hpo_test",
        base_task_project="examples",
        base_task_name="HPO: E-commerce Recommendation System",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/base_train_task_id": "your_step3_task_id_here",  # 需要替换为实际的task ID
            "General/num_trials": 2,
            "General/time_limit_minutes": 10,
            "General/test_queue": EXECUTION_QUEUE
        }
    )
    
    logger.info("Starting HPO test...")
    pipe.start_locally()
    logger.info("✅ HPO test started")


if __name__ == "__main__":
    """
    直接运行此文件进行测试
    """
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "simple":
            run_simple_pipeline()
        elif mode == "hpo":
            run_hpo_test()
        elif mode == "full":
            run_pipeline()
        else:
            print("Usage: python pipeline_from_tasks_complete.py [simple|hpo|full]")
    else:
        # 默认运行完整pipeline
        run_pipeline()
