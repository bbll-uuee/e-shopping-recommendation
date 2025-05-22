import argparse
from clearml import Task, OutputModel
from step3_train_model import train_model  # 假设你已有这个函数
import joblib
import os

# 初始化 ClearML 任务
task = Task.init(project_name="MyProject-HPO", task_name="Final-Model-Training")

# 设置命令行参数（也可以写死最优参数）
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=10)
args = parser.parse_args()

# 封装参数为 dict
params = {
    'learning_rate': args.learning_rate,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
    'save_model': True,
    'model_path': 'best_model.pkl',
}

# 执行训练（你需保证 train_model 支持这些参数）
best_score, model = train_model(params)

# 保存模型
os.makedirs('models', exist_ok=True)
model_path = os.path.join('models', 'best_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# 上传模型到 ClearML
output_model = OutputModel(task=task, framework="sklearn")
output_model.update_weights(weights_filename=model_path)
output_model.publish()
print(f"Model published: {output_model.id}")

task.close()
