from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, DiscreteParameterRange
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize HPO Task
task = Task.init(
    project_name='ecommerce_recommendation',
    task_name='HPO: Cluster & Recommend',
    task_type=Task.TaskTypes.optimizer
)

# Recommended: use an existing queue name
# Change 'pipeline' to your actual ClearML agent queue name
task.execute_remotely(queue_name='pipeline')

# Define optimizer
optimizer = HyperParameterOptimizer(
    base_task_id='',  
    hyper_parameters=[
        UniformIntegerParameterRange('General/n_clusters', 3, 10),
        UniformIntegerParameterRange('General/max_iter', 100, 500),
        DiscreteParameterRange('General/similarity_metric', ['cosine', 'euclidean']),
        UniformIntegerParameterRange('General/min_interactions', 1, 10),
    ],
    objective_metric_title='Clustering/Silhouette Score',
    objective_metric_series='Clustering/Silhouette Score',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=1,
    total_max_jobs=10,
    compute_time_limit=60 * 10,  # 10 minutes per trial
    min_iteration_per_job=1,
    max_iteration_per_job=1,
)

optimizer.set_time_limit(in_minutes=60)
optimizer.start()
optimizer.wait()

# Use get_parameters instead of hyperparameters
top_exp = optimizer.get_top_experiments(top_k=1)
if top_exp:
    best_params = top_exp[0].get_parameters()
    logger.info(f"Best configuration: {best_params}")
else:
    logger.warning("No successful experiments found.")
