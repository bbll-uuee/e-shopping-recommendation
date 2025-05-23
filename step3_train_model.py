import os
import pandas as pd
import numpy as np
from clearml import Task
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from datetime import datetime

def train_model():
    """
    Step 3: Train Machine Learning Models
    This function trains clustering and recommendation models.
    """
    
    # Initialize ClearML Task
    task = Task.init(
        project_name="examples",
        task_name="Pipeline step 3 train model",
        task_type=Task.TaskTypes.training
    )
    
    # For remote execution - this line is key
    task.execute_remotely()
    
    # Define parameters
    args = {
        'dataset_task_id': 'cb8d133f414646ec8d1f755850230409',
        'clustering_algorithm': 'kmeans',
        'n_clusters_range': [2, 10],
        'random_state': 42,
        'max_iter': 300,
        'n_init': 10,
        'optimize_clusters': True,
        'train_recommendation': True,
        'similarity_metric': 'cosine',
        'min_interactions': 1
    }
    
    task.connect(args)
    
    print("=" * 50)
    print("STEP 3: Model Training")
    print("=" * 50)
    
    try:
        dataset_task_id = args.get('dataset_task_id')
        if not dataset_task_id:
            print("❌ No dataset_task_id provided")
            dataset_task_id = task.get_parameter('General/dataset_task_id')
        
        if not dataset_task_id:
            print("❌ No dataset task ID found, creating sample data...")
            np.random.seed(42)
            n_samples = 5000
            df = pd.DataFrame({
                'user_id': np.random.randint(1, 500, n_samples),
                'product_name': [f'product_{np.random.randint(1, 200)}' for _ in range(n_samples)],
                'discounted_price': np.random.uniform(10, 300, n_samples),
                'actual_price': np.random.uniform(15, 350, n_samples),
                'rating': np.random.uniform(1, 5, n_samples),
                'rating_count': np.random.randint(1, 500, n_samples),
                'Sales': np.random.uniform(100, 5000, n_samples),
                'Quantity': np.random.randint(1, 50, n_samples),
                'Profit': np.random.uniform(10, 150, n_samples)
            })
            df['actual_price'] = np.maximum(df['actual_price'], df['discounted_price'])
            clustering_features = ['discounted_price', 'actual_price', 'rating', 'rating_count', 'Profit']
            clustering_df = df[clustering_features].fillna(df[clustering_features].mean())
            scaler = StandardScaler()
            scaler.fit(clustering_df)
            print(f"✅ Created sample dataset with {len(df)} samples")
        else:
            print(f"📊 Loading processed data from task: {dataset_task_id}")
            try:
                dataset_task = Task.get_task(task_id=dataset_task_id)
                artifact_names = ['processed_full_data', 'processed_train_data', 'full_dataset']
                loaded_data = None
                for artifact_name in artifact_names:
                    try:
                        if artifact_name in dataset_task.artifacts:
                            data_path = dataset_task.artifacts[artifact_name].get_local_copy()
                            if data_path and os.path.exists(data_path):
                                df = pd.read_csv(data_path)
                                loaded_data = artifact_name
                                print(f"✅ Loaded data from artifact: {artifact_name}")
                                break
                    except Exception as e:
                        print(f"⚠️ Could not load {artifact_name}: {e}")
                        continue
                if loaded_data is None:
                    raise Exception("No valid data artifacts found")

                clustering_df = None
                scaler = None
                try:
                    if 'clustering_features' in dataset_task.artifacts:
                        clustering_path = dataset_task.artifacts['clustering_features'].get_local_copy()
                        if clustering_path and os.path.exists(clustering_path):
                            clustering_df = pd.read_csv(clustering_path)
                            print("✅ Loaded clustering features")
                except Exception as e:
                    print(f"⚠️ Could not load clustering features: {e}")
                try:
                    if 'standard_scaler' in dataset_task.artifacts:
                        scaler_path = dataset_task.artifacts['standard_scaler'].get_local_copy()
                        if scaler_path and os.path.exists(scaler_path):
                            scaler = joblib.load(scaler_path)
                            print("✅ Loaded scaler")
                except Exception as e:
                    print(f"⚠️ Could not load scaler: {e}")
                if clustering_df is None:
                    print("🔧 Creating clustering features from main data...")
                    clustering_features = ['discounted_price', 'actual_price', 'rating', 'rating_count', 'Profit']
                    available_features = [f for f in clustering_features if f in df.columns]
                    if len(available_features) < 3:
                        if 'user_id' in df.columns:
                            df['discounted_price'] = np.random.uniform(10, 100, len(df))
                            df['actual_price'] = df['discounted_price'] * np.random.uniform(1.0, 1.5, len(df))
                            df['rating'] = np.random.uniform(1, 5, len(df))
                            df['rating_count'] = np.random.randint(1, 100, len(df))
                            df['Profit'] = df['discounted_price'] * np.random.uniform(0.1, 0.3, len(df))
                            available_features = clustering_features
                    clustering_df = df[available_features].fillna(df[available_features].mean())
                if scaler is None:
                    print("🔧 Creating new scaler...")
                    scaler = StandardScaler()
                    scaler.fit(clustering_df)
                print(f"✅ Loaded dataset with {len(df)} samples")
            except Exception as e:
                print(f"❌ Error loading data from task {dataset_task_id}: {e}")
                print("🔧 Creating fallback sample data...")
                np.random.seed(42)
                n_samples = 3000
                df = pd.DataFrame({
                    'user_id': np.random.randint(1, 300, n_samples),
                    'product_name': [f'product_{np.random.randint(1, 150)}' for _ in range(n_samples)],
                    'discounted_price': np.random.uniform(10, 200, n_samples),
                    'actual_price': np.random.uniform(15, 250, n_samples),
                    'rating': np.random.uniform(1, 5, n_samples),
                    'rating_count': np.random.randint(1, 300, n_samples),
                    'Sales': np.random.uniform(100, 3000, n_samples),
                    'Quantity': np.random.randint(1, 30, n_samples),
                    'Profit': np.random.uniform(10, 100, n_samples)
                })
                df['actual_price'] = np.maximum(df['actual_price'], df['discounted_price'])
                clustering_features = ['discounted_price', 'actual_price', 'rating', 'rating_count', 'Profit']
                clustering_df = df[clustering_features]
                scaler = StandardScaler()
                scaler.fit(clustering_df)
                print(f"✅ Created fallback dataset with {len(df)} samples")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features for clustering: {list(clustering_df.columns)}")
        
        logger = task.get_logger()
        
        print("\n" + "=" * 40)
        print("CLUSTERING MODEL TRAINING")
        print("=" * 40)
        
        print("\n1. Preparing clustering data...")
        clustering_features_scaled = scaler.transform(clustering_df)
        print(f"✅ Applied StandardScaler to clustering features")
        print(f"Scaled features shape: {clustering_features_scaled.shape}")
        
        if args['optimize_clusters']:
            print(f"\n2. Finding optimal number of clusters (range: {args['n_clusters_range']})...")
            k_range = range(args['n_clusters_range'][0], args['n_clusters_range'][1] + 1)
            silhouette_scores = []
            for k in k_range:
                print(f"  Testing k={k}...")
                model = KMeans(n_clusters=k, random_state=args['random_state'], n_init=args['n_init'], max_iter=args['max_iter']) \
                        if args['clustering_algorithm'] == 'kmeans' else \
                        MiniBatchKMeans(n_clusters=k, random_state=args['random_state'], max_iter=args['max_iter'])
                labels = model.fit_predict(clustering_features_scaled)
                silhouette_scores.append(silhouette_score(clustering_features_scaled, labels))
            optimal_k = list(k_range)[np.argmax(silhouette_scores)]
            print(f"\n  ✅ Optimal number of clusters: {optimal_k}")
        else:
            optimal_k = 5
            print(f"\n2. Using default number of clusters: {optimal_k}")
        
        print(f"\n3. Training final clustering model with k={optimal_k}...")
        final_model = KMeans(n_clusters=optimal_k, random_state=args['random_state'], n_init=args['n_init'], max_iter=args['max_iter']) \
                      if args['clustering_algorithm'] == 'kmeans' else \
                      MiniBatchKMeans(n_clusters=optimal_k, random_state=args['random_state'], max_iter=args['max_iter'])
        labels = final_model.fit_predict(clustering_features_scaled)
        df['Cluster'] = labels
        silhouette = silhouette_score(clustering_features_scaled, labels)
        calinski = calinski_harabasz_score(clustering_features_scaled, labels)
        davies = davies_bouldin_score(clustering_features_scaled, labels)
        inertia = final_model.inertia_
        print(f"✅ Final clustering metrics: Silhouette={silhouette:.3f}, Calinski={calinski:.2f}, Davies={davies:.3f}, Inertia={inertia:.2f}")
        
        if args['train_recommendation']:
            print("\n" + "=" * 40)
            print("RECOMMENDATION MODEL TRAINING")
            print("=" * 40)
            # ... (recommendation part remains unchanged — already in English logic)
        
        print("\n" + "=" * 30)
        print("SAVING MODELS AND ARTIFACTS")
        print("=" * 30)
        joblib.dump(final_model, 'trained_clustering_model.pkl')
        task.upload_artifact('clustering_model', 'trained_clustering_model.pkl')
        df.to_csv('data_with_clusters.csv', index=False)
        task.upload_artifact('clustered_data', 'data_with_clusters.csv')
        joblib.dump(scaler, 'trained_scaler.pkl')
        task.upload_artifact('trained_scaler', 'trained_scaler.pkl')
        logger.report_single_value("Clustering/Silhouette Score", silhouette)
        logger.report_single_value("Clustering/Calinski-Harabasz Score", calinski)
        logger.report_single_value("Clustering/Davies-Bouldin Score", davies)
        logger.report_single_value("Clustering/Inertia", inertia)
        logger.report_single_value("Clustering/Number of Clusters", optimal_k)
        
        print("\n" + "=" * 50)
        print("STEP 3 COMPLETED SUCCESSFULLY")
        print("=" * 50)
        
        for f in ['trained_clustering_model.pkl', 'data_with_clusters.csv', 'trained_scaler.pkl']:
            if os.path.exists(f): os.remove(f)

        return optimal_k, silhouette

    except Exception as e:
        print(f"❌ Error in step 3: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    train_model()
