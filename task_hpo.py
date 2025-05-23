from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange, DiscreteParameterRange
import logging
import time
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(
    project_name='examples',  # Match your ClearML project name
    task_name='HPO: E-commerce Recommendation System',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Connect parameters - adapted for e-commerce recommendation system
args = {
    'base_train_task_id': '719e41ed1ff942b79778bfac03e7e121',  # This will be set by the pipeline
    'num_trials': 3,  # Reduced from 10 for faster testing
    'time_limit_minutes': 20,  # Reduced from 60
    'run_as_service': False,
    'test_queue': 'pipeline',
    'processed_dataset_id': '',  # Provided by pipeline

    # E-commerce parameters (analogous to DL parameters)
    'n_clusters_max': 10,
    'max_iter': 300,
    'outlier_threshold': 3.0,
    'min_interactions': 2
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID
dataset_id = task.get_parameter('General/processed_dataset_id')
if not dataset_id:
    dataset_id = args.get('processed_dataset_id')
    logger.info(f"No dataset_id in General namespace, using from args: {dataset_id}")
if not dataset_id:
    dataset_id = "your_dataset_id_here"
    logger.info(f"Using fallback dataset ID: {dataset_id}")

logger.info(f"Using dataset ID: {dataset_id}")

# Get base training task ID
try:
    BASE_TRAIN_TASK_ID = args['base_train_task_id']
    logger.info(f"Using base training task ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task ID: {e}")
    raise

logger.info("Dataset verification skipped (non-ClearML Dataset)")

# Define HPO task
hpo_task = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        UniformIntegerParameterRange('n_clusters_range_max', 3, args['n_clusters_max']),
        UniformIntegerParameterRange('max_iter', 100, 500),
        UniformParameterRange('outlier_threshold', 2.0, 4.0),
        UniformIntegerParameterRange('min_interactions', 1, 5),
        DiscreteParameterRange('clustering_algorithm', ['kmeans', 'minibatch_kmeans']),
        DiscreteParameterRange('similarity_metric', ['cosine', 'euclidean'])
    ],
    objective_metric_title='Clustering',
    objective_metric_series='Silhouette Score',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=1,
    pool_period_min=1.0,
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,
    parameter_override={
        'processed_dataset_id': dataset_id,
        'General/processed_dataset_id': dataset_id,
        'test_queue': args['test_queue'],
        'General/test_queue': args['test_queue'],
        'n_clusters_max': args['n_clusters_max'],
        'General/n_clusters_max': args['n_clusters_max'],
        'max_iter': args['max_iter'],
        'General/max_iter': args['max_iter'],
        'outlier_threshold': args['outlier_threshold'],
        'General/outlier_threshold': args['outlier_threshold'],
        'min_interactions': args['min_interactions'],
        'General/min_interactions': args['min_interactions'],
        'optimize_clusters': True,
        'General/optimize_clusters': True,
        'train_recommendation': True,
        'General/train_recommendation': True
    }
)

# Start HPO
logger.info("Starting HPO task...")
hpo_task.start()

logger.info(f"Waiting for optimization to complete (time limit: {args['time_limit_minutes']} minutes)...")
time.sleep(args['time_limit_minutes'] * 60)

try:
    top_exp = hpo_task.get_top_experiments(top_k=3)
    if top_exp:
        best_exp = top_exp[0]
        logger.info(f"Best experiment: {best_exp.id}")

        best_params = best_exp.get_parameters()
        metrics = best_exp.get_last_scalar_metrics()

        best_silhouette = None
        best_calinski = None
        recommendation_users = None

        if metrics:
            clustering_metrics = metrics.get('Clustering', {})
            best_silhouette = clustering_metrics.get('Silhouette Score')
            best_calinski = clustering_metrics.get('Calinski-Harabasz Score')
            recommendation_metrics = metrics.get('Recommendation', {})
            recommendation_users = recommendation_metrics.get('Unique Users')

        logger.info("Best experiment parameters:")
        for k, v in best_params.items():
            logger.info(f"  - {k}: {v}")
        logger.info(f"Best silhouette score: {best_silhouette}")
        logger.info(f"Best calinski score: {best_calinski}")
        logger.info(f"Recommendation users: {recommendation_users}")

        best_results = {
            'parameters': best_params,
            'silhouette_score': best_silhouette,
            'calinski_score': best_calinski,
            'recommendation_users': recommendation_users,
            'experiment_id': best_exp.id
        }

        all_experiments = []
        for i, exp in enumerate(top_exp):
            exp_params = exp.get_parameters()
            exp_metrics = exp.get_last_scalar_metrics()
            silhouette = exp_metrics.get('Clustering', {}).get('Silhouette Score')
            all_experiments.append({
                'rank': i + 1,
                'experiment_id': exp.id,
                'parameters': exp_params,
                'silhouette_score': silhouette
            })

        with open('best_parameters_ecommerce.json', 'w') as f:
            json.dump({
                'best_experiment': best_results,
                'all_experiments': all_experiments,
                'optimization_summary': {
                    'target_metric': 'Silhouette Score',
                    'optimization_type': 'maximize',
                    'total_trials': args['num_trials'],
                    'time_limit_minutes': args['time_limit_minutes']
                }
            }, f, indent=4, default=str)
        task.upload_artifact('best_parameters', 'best_parameters_ecommerce.json')

        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_silhouette_score', best_silhouette)
        task.set_parameter('best_experiment_id', best_exp.id)
        task.set_parameter('best_clustering_algorithm', best_params.get('clustering_algorithm'))
        task.set_parameter('best_n_clusters', best_params.get('n_clusters_range_max'))

        os.remove('best_parameters_ecommerce.json')

    else:
        logger.warning("No experiments completed. Possible reasons include:")
        logger.warning("  - Step 3 tasks are slow or failed")
        logger.warning("  - No workers available in 'pipeline' queue")
        logger.warning("  - Dataset misconfiguration")

        incomplete_info = {
            'status': 'no_completed_experiments',
            'base_task_id': BASE_TRAIN_TASK_ID,
            'time_limit_minutes': args['time_limit_minutes'],
            'suggestions': [
                'Check if step3 task finished correctly',
                'Ensure workers are active in the pipeline queue',
                'Increase the time limit if needed',
                'Verify the dataset ID is correct'
            ]
        }

        with open('hpo_incomplete_status.json', 'w') as f:
            json.dump(incomplete_info, f, indent=4)
        task.upload_artifact('hpo_status', 'hpo_incomplete_status.json')
        os.remove('hpo_incomplete_status.json')

except Exception as e:
    logger.error(f"Error fetching experiments: {e}")
    error_details = {
        'error_message': str(e),
        'error_type': type(e).__name__,
        'base_task_id': BASE_TRAIN_TASK_ID,
        'dataset_id': dataset_id,
        'project_type': 'e-commerce_recommendation_system'
    }
    with open('hpo_error_details.json', 'w') as f:
        json.dump(error_details, f, indent=4)
    task.upload_artifact('hpo_error', 'hpo_error_details.json')
    os.remove('hpo_error_details.json')
    raise

# Stop the HPO process
hpo_task.stop()
logger.info("Optimizer stopped")

logger.info("=" * 60)
logger.info("E-COMMERCE RECOMMENDATION SYSTEM HPO COMPLETED")
logger.info("=" * 60)
logger.info("Check ClearML UI for:")
logger.info("  - Best experiment parameters")
logger.info("  - Clustering performance metrics")
logger.info("  - Recommendation system results")
logger.info("Use best parameters in final_model.py for deployment")

print('E-commerce HPO completed - ready for final model!')
