from clearml import Task
from clearml.automation import UniformParameterRange, DiscreteParameterRange, HyperParameterOptimizer

# Initialize the ClearML HPO task
task = Task.init(
    project_name="MyProject-HPO",
    task_name="Optuna-HPO",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)

# This is the base task to clone for each HPO trial (must support arguments like --learning_rate etc.)
base_task_id = 'INSERT_YOUR_BASE_TRAIN_TASK_ID_HERE'

# Define the hyperparameter search space
param_ranges = {
    'Args/learning_rate': UniformParameterRange(1e-5, 1e-2),
    'Args/batch_size': DiscreteParameterRange([16, 32, 64]),
    'Args/num_epochs': DiscreteParameterRange([5, 10, 20]),
}

# Create the HPO optimizer
optimizer = HyperParameterOptimizer(
    base_task_id=base_task_id,  # the experiment to use as base
    hyper_parameters=param_ranges,
    objective_metric_title='validation',
    objective_metric_series='accuracy',
    objective_metric_sign='max',  # maximize accuracy
    max_number_of_concurrent_tasks=2,
    optimizer_class='BayesianOptimization',
    execution_queue='default',
    max_iteration=20,  # total number of HPO trials
)

# Start the optimization
optimizer.start()

# Print the best result
top_exp = optimizer.get_best_top_experiments(top_k=1)[0]
print("Best configuration:")
print(top_exp.hyper_parameters)
print("Best score:", top_exp.metrics['validation']['accuracy'])

# Optionally stop the task
task.close()
