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
    project_name='examples',  # 匹配您的项目名称
    task_name='HPO: E-commerce Recommendation System',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Connect parameters - 完全仿照老师的结构，但适配电商推荐系统
args = {
    'base_train_task_id': '6da1e806d41d419cb6c71fd1f683ca58',  # Will be set from pipeline - 您的step3任务ID
    'num_trials': 3,  # Reduced from 10 to 3 trials
    'time_limit_minutes': 20,  # Reduced from 60 to 20 minutes
    'run_as_service': False,
    'test_queue': 'pipeline',  # Queue for test tasks
    'processed_dataset_id': '',  # Will be set from pipeline - 您的数据集ID
    
    # 电商推荐系统参数 - 对应老师例子中的深度学习参数
    'n_clusters_max': 10,  # 对应 num_epochs - 最大聚类数
    'max_iter': 300,  # 对应 batch_size - 聚类最大迭代次数
    'outlier_threshold': 3.0,  # 对应 learning_rate - 异常值阈值
    'min_interactions': 2  # 对应 weight_decay - 最小交互次数
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID from pipeline parameters
dataset_id = task.get_parameter('General/processed_dataset_id')  # Get from General namespace
if not dataset_id:
    # Try getting from args as fallback
    dataset_id = args.get('processed_dataset_id')
    logger.info(f"No dataset_id in General namespace, using from args: {dataset_id}")

if not dataset_id:
    # Use fixed dataset ID as last resort - 您需要替换这个ID
    dataset_id = "your_dataset_id_here"
    logger.info(f"Using fixed dataset ID: {dataset_id}")

logger.info(f"Using dataset ID: {dataset_id}")

# Get the actual training model task
try:
    BASE_TRAIN_TASK_ID = args['base_train_task_id']
    logger.info(f"Using base training task ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task ID: {e}")
    raise

# Verify dataset exists - 注释掉数据集验证，因为我们的数据集可能不是ClearML Dataset格式
# try:
#     dataset = Dataset.get(dataset_id=dataset_id)
#     logger.info(f"Successfully verified dataset: {dataset.name}")
# except Exception as e:
#     logger.error(f"Failed to verify dataset: {e}")
#     raise

logger.info("Dataset verification skipped for e-commerce project")

# Create the HPO task - 完全仿照老师的结构
hpo_task = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        # 聚类参数 - 对应老师例子中的训练参数
        UniformIntegerParameterRange('n_clusters_range_max', min_value=3, max_value=args['n_clusters_max']),
        UniformIntegerParameterRange('max_iter', min_value=100, max_value=500),  # 对应 batch_size 范围
        UniformParameterRange('outlier_threshold', min_value=2.0, max_value=4.0),  # 对应 learning_rate 范围
        UniformIntegerParameterRange('min_interactions', min_value=1, max_value=5),  # 对应 weight_decay 范围
        
        # 算法选择参数
        DiscreteParameterRange('clustering_algorithm', values=['kmeans', 'minibatch_kmeans']),
        DiscreteParameterRange('similarity_metric', values=['cosine', 'euclidean'])
    ],
    
    # 目标指标 - 改为聚类相关指标
    objective_metric_title='Clustering',  # 对应老师的 'validation'
    objective_metric_series='Silhouette Score',  # 对应老师的 'accuracy'
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    compute_time_limit=None,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=1,  # 聚类任务每次只运行一轮
    pool_period_min=1.0,  # Reduced from 2.0 to 1.0 to check more frequently
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,  # Reduced from 5 to 2
    
    # 参数覆盖 - 完全仿照老师的详细设置
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
        # 确保关键开关开启
        'optimize_clusters': True,
        'General/optimize_clusters': True,
        'train_recommendation': True,
        'General/train_recommendation': True
    }
)

# Start the HPO task
logger.info("Starting HPO task...")
hpo_task.start()

# Wait for optimization to complete
logger.info(f"Waiting for optimization to complete (time limit: {args['time_limit_minutes']} minutes)...")
time.sleep(args['time_limit_minutes'] * 60)  # Wait for the full time limit

# Get the top performing experiments - 完全仿照老师的结果处理
try:
    top_exp = hpo_task.get_top_experiments(top_k=3)  # Get top 3 experiments
    if top_exp:
        best_exp = top_exp[0]
        logger.info(f"Best experiment: {best_exp.id}")
        
        # Get the best parameters and metrics - 适配电商推荐系统的指标
        best_params = best_exp.get_parameters()
        metrics = best_exp.get_last_scalar_metrics()
        
        # 提取电商系统的关键指标
        best_silhouette = None
        best_calinski = None
        recommendation_users = None
        
        if metrics:
            if 'Clustering' in metrics:
                clustering_metrics = metrics['Clustering']
                best_silhouette = clustering_metrics.get('Silhouette Score')
                best_calinski = clustering_metrics.get('Calinski-Harabasz Score')
            
            if 'Recommendation' in metrics:
                rec_metrics = metrics['Recommendation']
                recommendation_users = rec_metrics.get('Unique Users')
        
        # Log detailed information about the best experiment - 完全仿照老师的格式
        logger.info("Best experiment parameters:")
        logger.info(f"  - clustering_algorithm: {best_params.get('clustering_algorithm')}")
        logger.info(f"  - n_clusters_range_max: {best_params.get('n_clusters_range_max')}")
        logger.info(f"  - max_iter: {best_params.get('max_iter')}")
        logger.info(f"  - outlier_threshold: {best_params.get('outlier_threshold')}")
        logger.info(f"  - similarity_metric: {best_params.get('similarity_metric')}")
        logger.info(f"  - min_interactions: {best_params.get('min_interactions')}")
        logger.info(f"Best silhouette score: {best_silhouette}")
        logger.info(f"Best calinski score: {best_calinski}")
        logger.info(f"Recommendation users: {recommendation_users}")
        
        # Save best parameters and metrics - 完全仿照老师的保存方式
        best_results = {
            'parameters': best_params,
            'silhouette_score': best_silhouette,
            'calinski_score': best_calinski,
            'recommendation_users': recommendation_users,
            'experiment_id': best_exp.id
        }
        
        # 处理所有top实验
        all_experiments = []
        for i, exp in enumerate(top_exp):
            exp_params = exp.get_parameters()
            exp_metrics = exp.get_last_scalar_metrics()
            
            exp_silhouette = None
            if exp_metrics and 'Clustering' in exp_metrics:
                exp_silhouette = exp_metrics['Clustering'].get('Silhouette Score')
            
            exp_info = {
                'rank': i + 1,
                'experiment_id': exp.id,
                'parameters': exp_params,
                'silhouette_score': exp_silhouette
            }
            all_experiments.append(exp_info)
            
            logger.info(f"Experiment {i+1}: ID={exp.id}, Silhouette={exp_silhouette}")
        
        # Save to a temporary file - 完全仿照老师的方式
        temp_file = 'best_parameters_ecommerce.json'
        with open(temp_file, 'w') as f:
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
        
        # Upload as artifact
        task.upload_artifact('best_parameters', temp_file)
        logger.info(f"Saved best parameters with silhouette score: {best_silhouette}")
        
        # Also save as task parameters for easier access - 完全仿照老师的方式
        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_silhouette_score', best_silhouette)
        task.set_parameter('best_experiment_id', best_exp.id)
        task.set_parameter('best_clustering_algorithm', best_params.get('clustering_algorithm'))
        task.set_parameter('best_n_clusters', best_params.get('n_clusters_range_max'))
        
        logger.info("Best parameters saved as both artifact and task parameters")
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    else:
        logger.warning("No experiments completed yet. This might be normal if the optimization just started.")
        logger.warning("Possible reasons for e-commerce project:")
        logger.warning("  - Clustering tasks take longer than expected")
        logger.warning("  - Base step3 task is still running")
        logger.warning("  - Worker queue 'pipeline' not available")
        logger.warning("  - Dataset loading issues")
        
        # 保存未完成状态
        incomplete_info = {
            'status': 'no_completed_experiments',
            'base_task_id': BASE_TRAIN_TASK_ID,
            'time_limit_minutes': args['time_limit_minutes'],
            'suggestions': [
                'Check if base task (step3) completes successfully',
                'Verify pipeline queue has active workers',
                'Consider increasing time_limit_minutes',
                'Check dataset_id is correct'
            ]
        }
        
        temp_status_file = 'hpo_incomplete_status.json'
        with open(temp_status_file, 'w') as f:
            json.dump(incomplete_info, f, indent=4)
        
        task.upload_artifact('hpo_status', temp_status_file)
        
        if os.path.exists(temp_status_file):
            os.remove(temp_status_file)

except Exception as e:
    logger.error(f"Failed to get top experiments: {e}")
    
    # 保存错误信息 - 仿照老师的错误处理
    error_details = {
        'error_message': str(e),
        'error_type': type(e).__name__,
        'base_task_id': BASE_TRAIN_TASK_ID,
        'dataset_id': dataset_id,
        'project_type': 'e-commerce_recommendation_system'
    }
    
    temp_error_file = 'hpo_error_details.json'
    with open(temp_error_file, 'w') as f:
        json.dump(error_details, f, indent=4)
    
    task.upload_artifact('hpo_error', temp_error_file)
    
    if os.path.exists(temp_error_file):
        os.remove(temp_error_file)
    
    raise

# Make sure background optimization stopped
hpo_task.stop()
logger.info("Optimizer stopped")

# 最终消息 - 仿照老师的结束方式
logger.info("="*60)
logger.info("E-COMMERCE RECOMMENDATION SYSTEM HPO COMPLETED")
logger.info("="*60)
logger.info("Check ClearML UI for:")
logger.info("  - Best experiment parameters")
logger.info("  - Clustering performance metrics")
logger.info("  - Recommendation system results")
logger.info("Use best parameters in final_model.py for deployment")

print('E-commerce HPO completed - ready for final model!')
