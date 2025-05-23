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
    This function trains clustering and recommendation models
    """
    
    # Initialize ClearML Task
    task = Task.init(
        project_name="examples",
        task_name="Pipeline step 3 train model",
        task_type=Task.TaskTypes.training
    )
    
    # For remote execution - this line is key
    task.execute_remotely()
    
    # Get parameters from task (can be overridden from pipeline)
    args = {
        'dataset_task_id': '',  # Will be overridden by pipeline
        'clustering_algorithm': 'kmeans',  # 'kmeans' or 'minibatch_kmeans'
        'n_clusters_range': [2, 10],  # Range for optimal cluster search
        'random_state': 42,
        'max_iter': 300,
        'n_init': 10,
        'optimize_clusters': True,
        'train_recommendation': True,
        'similarity_metric': 'cosine',  # 'cosine' or 'euclidean'
        'min_interactions': 1  # Minimum interactions for recommendation
    }
    
    # Connect parameters to task
    task.connect(args)
    
    print("=" * 50)
    print("STEP 3: Model Training")
    print("=" * 50)
    
    try:
        # Get processed data from previous step
        if args['dataset_task_id']:
            print(f"Loading processed data from task: {args['dataset_task_id']}")
            dataset_task = Task.get_task(task_id=args['dataset_task_id'])
            
            # Download artifacts from preprocessing step
            processed_full_data = dataset_task.artifacts['processed_full_data'].get_local_copy()
            clustering_features_data = dataset_task.artifacts['clustering_features'].get_local_copy()
            scaler_artifact = dataset_task.artifacts['standard_scaler'].get_local_copy()
            metadata = dataset_task.artifacts['preprocessing_metadata'].get()
            
            # Load data
            df = pd.read_csv(processed_full_data)
            clustering_df = pd.read_csv(clustering_features_data)
            scaler = joblib.load(scaler_artifact)
            
            print(f"Loaded processed dataset with {len(df)} samples")
            print(f"Clustering features shape: {clustering_df.shape}")
            
        else:
            raise ValueError("dataset_task_id not provided")
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features for clustering: {list(clustering_df.columns)}")
        
        # Initialize logger
        logger = task.get_logger()
        
        # PART 1: CLUSTERING MODEL TRAINING
        print("\n" + "=" * 40)
        print("PART 1: CLUSTERING MODEL TRAINING")
        print("=" * 40)
        
        # Prepare clustering data
        print("\n1. Preparing clustering data...")
        
        # Use the scaler from preprocessing step
        clustering_features_scaled = scaler.transform(clustering_df)
        print(f"Applied StandardScaler to clustering features")
        print(f"Scaled features shape: {clustering_features_scaled.shape}")
        
        clustering_results = {}
        
        # 2. Find optimal number of clusters
        if args['optimize_clusters']:
            print(f"\n2. Finding optimal number of clusters (range: {args['n_clusters_range']})...")
            
            k_range = range(args['n_clusters_range'][0], args['n_clusters_range'][1] + 1)
            inertia_scores = []
            silhouette_scores = []
            calinski_scores = []
            davies_bouldin_scores = []
            
            # Test different numbers of clusters
            for k in k_range:
                print(f"  Testing k={k}...")
                
                if args['clustering_algorithm'] == 'kmeans':
                    kmeans = KMeans(
                        n_clusters=k,
                        random_state=args['random_state'],
                        n_init=args['n_init'],
                        max_iter=args['max_iter']
                    )
                else:  # minibatch_kmeans
                    kmeans = MiniBatchKMeans(
                        n_clusters=k,
                        random_state=args['random_state'],
                        max_iter=args['max_iter']
                    )
                
                # Fit the model
                cluster_labels = kmeans.fit_predict(clustering_features_scaled)
                
                # Calculate metrics
                inertia = kmeans.inertia_
                silhouette = silhouette_score(clustering_features_scaled, cluster_labels)
                calinski = calinski_harabasz_score(clustering_features_scaled, cluster_labels)
                davies_bouldin = davies_bouldin_score(clustering_features_scaled, cluster_labels)
                
                inertia_scores.append(inertia)
                silhouette_scores.append(silhouette)
                calinski_scores.append(calinski)
                davies_bouldin_scores.append(davies_bouldin)
                
                print(f"    Inertia: {inertia:.2f}, Silhouette: {silhouette:.3f}, "
                      f"Calinski-Harabasz: {calinski:.2f}, Davies-Bouldin: {davies_bouldin:.3f}")
            
            # Store optimization results
            clustering_results['optimization'] = {
                'k_range': list(k_range),
                'inertia_scores': inertia_scores,
                'silhouette_scores': silhouette_scores,
                'calinski_scores': calinski_scores,
                'davies_bouldin_scores': davies_bouldin_scores
            }
            
            # Find optimal k (highest silhouette score)
            optimal_k_idx = np.argmax(silhouette_scores)
            optimal_k = list(k_range)[optimal_k_idx]
            optimal_silhouette = silhouette_scores[optimal_k_idx]
            
            print(f"\n  Optimal number of clusters: {optimal_k}")
            print(f"  Best silhouette score: {optimal_silhouette:.3f}")
            
            # Visualize optimization results
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Elbow method
            axes[0, 0].plot(k_range, inertia_scores, marker='o', linestyle='--')
            axes[0, 0].set_title('Elbow Method for Optimal K')
            axes[0, 0].set_xlabel('Number of Clusters (K)')
            axes[0, 0].set_ylabel('Inertia')
            axes[0, 0].grid(True)
            
            # Silhouette score
            axes[0, 1].plot(k_range, silhouette_scores, marker='o', linestyle='--', color='green')
            axes[0, 1].axvline(x=optimal_k, color='red', linestyle=':', label=f'Optimal K={optimal_k}')
            axes[0, 1].set_title('Silhouette Score vs Number of Clusters')
            axes[0, 1].set_xlabel('Number of Clusters (K)')
            axes[0, 1].set_ylabel('Silhouette Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Calinski-Harabasz score
            axes[1, 0].plot(k_range, calinski_scores, marker='o', linestyle='--', color='orange')
            axes[1, 0].set_title('Calinski-Harabasz Score vs Number of Clusters')
            axes[1, 0].set_xlabel('Number of Clusters (K)')
            axes[1, 0].set_ylabel('Calinski-Harabasz Score')
            axes[1, 0].grid(True)
            
            # Davies-Bouldin score (lower is better)
            axes[1, 1].plot(k_range, davies_bouldin_scores, marker='o', linestyle='--', color='purple')
            axes[1, 1].set_title('Davies-Bouldin Score vs Number of Clusters')
            axes[1, 1].set_xlabel('Number of Clusters (K)')
            axes[1, 1].set_ylabel('Davies-Bouldin Score')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            logger.report_matplotlib_figure(
                title="Clustering Optimization",
                series="Metrics vs K",
                figure=plt.gcf()
            )
            plt.close()
            
        else:
            optimal_k = 3  # Default value
            print(f"\n2. Using default number of clusters: {optimal_k}")
        
        # 3. Train final clustering model
        print(f"\n3. Training final clustering model with k={optimal_k}...")
        
        if args['clustering_algorithm'] == 'kmeans':
            final_kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=args['random_state'],
                n_init=args['n_init'],
                max_iter=args['max_iter']
            )
        else:
            final_kmeans = MiniBatchKMeans(
                n_clusters=optimal_k,
                random_state=args['random_state'],
                max_iter=args['max_iter']
            )
        
        # Fit the final model
        cluster_labels = final_kmeans.fit_predict(clustering_features_scaled)
        
        # Add cluster labels to original dataframe
        df['Cluster'] = cluster_labels
        
        # Calculate final metrics
        final_inertia = final_kmeans.inertia_
        final_silhouette = silhouette_score(clustering_features_scaled, cluster_labels)
        final_calinski = calinski_harabasz_score(clustering_features_scaled, cluster_labels)
        final_davies_bouldin = davies_bouldin_score(clustering_features_scaled, cluster_labels)
        
        print(f"Final clustering metrics:")
        print(f"  Inertia: {final_inertia:.2f}")
        print(f"  Silhouette Score: {final_silhouette:.3f}")
        print(f"  Calinski-Harabasz Score: {final_calinski:.2f}")
        print(f"  Davies-Bouldin Score: {final_davies_bouldin:.3f}")
        
        # 4. Analyze clusters
        print(f"\n4. Analyzing clusters...")
        
        cluster_summary = df.groupby('Cluster')[list(clustering_df.columns)].agg(['mean', 'std', 'count'])
        print("Cluster summary statistics:")
        print(cluster_summary)
        
        # Cluster sizes
        cluster_sizes = df['Cluster'].value_counts().sort_index()
        print(f"\nCluster sizes:")
        for i, size in enumerate(cluster_sizes):
            print(f"  Cluster {i}: {size} samples ({size/len(df)*100:.1f}%)")
        
        # 5. Visualize clustering results
        print(f"\n5. Generating clustering visualizations...")
        
        # Cluster visualization (2D projection using first two features)
        plt.figure(figsize=(12, 8))
        
        # Create subplot for different feature pairs
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        feature_pairs = [
            ('discounted_price', 'actual_price'),
            ('rating', 'Profit'),
            ('discounted_price', 'rating'),
            ('actual_price', 'Profit')
        ]
        
        for idx, (x_feature, y_feature) in enumerate(feature_pairs):
            if x_feature in df.columns and y_feature in df.columns:
                ax = axes[idx // 2, idx % 2]
                scatter = ax.scatter(df[x_feature], df[y_feature], 
                                   c=df['Cluster'], cmap='viridis', 
                                   alpha=0.6, s=50)
                ax.set_xlabel(x_feature.replace('_', ' ').title())
                ax.set_ylabel(y_feature.replace('_', ' ').title())
                ax.set_title(f'Clusters: {x_feature.replace("_", " ").title()} vs {y_feature.replace("_", " ").title()}')
                plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        logger.report_matplotlib_figure(
            title="Clustering Results",
            series="Feature Space Visualization",
            figure=plt.gcf()
        )
        plt.close()
        
        # Cluster distribution
        plt.figure(figsize=(10, 6))
        cluster_sizes.plot(kind='bar', alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3)
        logger.report_matplotlib_figure(
            title="Clustering Results",
            series="Cluster Distribution",
            figure=plt.gcf()
        )
        plt.close()
        
        # Store clustering results
        clustering_results['final_model'] = {
            'algorithm': args['clustering_algorithm'],
            'n_clusters': optimal_k,
            'inertia': final_inertia,
            'silhouette_score': final_silhouette,
            'calinski_harabasz_score': final_calinski,
            'davies_bouldin_score': final_davies_bouldin,
            'cluster_sizes': cluster_sizes.to_dict()
        }
        
        # PART 2: RECOMMENDATION MODEL TRAINING
        if args['train_recommendation']:
            print("\n" + "=" * 40)
            print("PART 2: RECOMMENDATION MODEL TRAINING")
            print("=" * 40)
            
            # 1. Prepare recommendation data
            print("\n1. Preparing recommendation data...")
            
            # Filter users and products with minimum interactions
            user_interactions = df.groupby('user_id').size()
            product_interactions = df.groupby('product_name').size()
            
            valid_users = user_interactions[user_interactions >= args['min_interactions']].index
            valid_products = product_interactions[product_interactions >= args['min_interactions']].index
            
            # Filter data
            recommendation_df = df[
                (df['user_id'].isin(valid_users)) & 
                (df['product_name'].isin(valid_products))
            ].copy()
            
            print(f"Recommendation data shape: {recommendation_df.shape}")
            print(f"Valid users: {len(valid_users)}")
            print(f"Valid products: {len(valid_products)}")
            
            # 2. Create user-product matrix
            print("\n2. Creating user-product interaction matrix...")
            
            # Aggregate quantities for each user-product pair
            user_product_interactions = recommendation_df.groupby(['user_id', 'product_name'])['Quantity'].sum().reset_index()
            
            # Create pivot table (user-product matrix)
            user_product_matrix = user_product_interactions.pivot(
                index='user_id', 
                columns='product_name', 
                values='Quantity'
            ).fillna(0)
            
            print(f"User-product matrix shape: {user_product_matrix.shape}")
            print(f"Matrix sparsity: {(user_product_matrix == 0).sum().sum() / (user_product_matrix.shape[0] * user_product_matrix.shape[1]) * 100:.2f}%")
            
            # 3. Calculate similarity matrices
            print(f"\n3. Calculating similarity matrices using {args['similarity_metric']} similarity...")
            
            if args['similarity_metric'] == 'cosine':
                # Product-product similarity
                product_similarity = cosine_similarity(user_product_matrix.T)
                # User-user similarity (for collaborative filtering)
                user_similarity = cosine_similarity(user_product_matrix)
            else:
                # For euclidean or other metrics, you might need different approaches
                from sklearn.metrics.pairwise import euclidean_distances
                product_similarity = 1 / (1 + euclidean_distances(user_product_matrix.T))
                user_similarity = 1 / (1 + euclidean_distances(user_product_matrix))
            
            # Convert to DataFrames for easier handling
            product_similarity_df = pd.DataFrame(
                product_similarity,
                index=user_product_matrix.columns,
                columns=user_product_matrix.columns
            )
            
            user_similarity_df = pd.DataFrame(
                user_similarity,
                index=user_product_matrix.index,
                columns=user_product_matrix.index
            )
            
            print(f"Product similarity matrix shape: {product_similarity_df.shape}")
            print(f"User similarity matrix shape: {user_similarity_df.shape}")
            
            # 4. Create recommendation functions
            print("\n4. Creating recommendation system...")
            
            def get_product_recommendations(product_name, top_n=5):
                """Get similar products based on product-product similarity"""
                product_name = product_name.strip().lower()
                
                if product_name not in product_similarity_df.columns:
                    return f"Product '{product_name}' not found in similarity matrix."
                
                sim_scores = product_similarity_df[product_name].sort_values(ascending=False)
                similar_products = sim_scores.drop(product_name).head(top_n)
                return similar_products
            
            def get_user_recommendations(user_id, top_n=5):
                """Get product recommendations for a user based on user-user similarity"""
                if user_id not in user_similarity_df.index:
                    return f"User {user_id} not found in similarity matrix."
                
                # Find similar users
                user_sim_scores = user_similarity_df.loc[user_id].sort_values(ascending=False)
                similar_users = user_sim_scores.drop(user_id).head(10).index  # Top 10 similar users
                
                # Get products that similar users liked but current user hasn't interacted with
                user_products = set(user_product_matrix.loc[user_id][user_product_matrix.loc[user_id] > 0].index)
                
                recommendations = {}
                for similar_user in similar_users:
                    similar_user_products = user_product_matrix.loc[similar_user]
                    for product, rating in similar_user_products.items():
                        if rating > 0 and product not in user_products:
                            if product not in recommendations:
                                recommendations[product] = 0
                            recommendations[product] += rating * user_sim_scores[similar_user]
                
                # Sort recommendations
                sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
                return sorted_recommendations[:top_n]
            
            # 5. Test recommendation system
            print("\n5. Testing recommendation system...")
            
            # Test product-based recommendations
            sample_products = list(product_similarity_df.columns)[:5]
            print("Sample product-based recommendations:")
            for product in sample_products:
                recs = get_product_recommendations(product, top_n=3)
                if isinstance(recs, pd.Series):
                    print(f"  Similar to '{product}': {list(recs.head(3).index)}")
            
            # Test user-based recommendations
            sample_users = list(user_similarity_df.index)[:3]
            print("\nSample user-based recommendations:")
            for user in sample_users:
                recs = get_user_recommendations(user, top_n=3)
                if isinstance(recs, list):
                    product_names = [rec[0] for rec in recs]
                    print(f"  For user {user}: {product_names}")
            
            # 6. Evaluate recommendation system
            print("\n6. Evaluating recommendation system...")
            
            # Calculate basic metrics
            total_interactions = len(recommendation_df)
            unique_users = len(valid_users)
            unique_products = len(valid_products)
            avg_interactions_per_user = total_interactions / unique_users
            avg_interactions_per_product = total_interactions / unique_products
            
            recommendation_metrics = {
                'total_interactions': total_interactions,
                'unique_users': unique_users,
                'unique_products': unique_products,
                'avg_interactions_per_user': avg_interactions_per_user,
                'avg_interactions_per_product': avg_interactions_per_product,
                'matrix_sparsity': (user_product_matrix == 0).sum().sum() / (user_product_matrix.shape[0] * user_product_matrix.shape[1]),
                'similarity_metric': args['similarity_metric']
            }
            
            print("Recommendation system metrics:")
            for metric, value in recommendation_metrics.items():
                print(f"  {metric}: {value}")
            
            # 7. Visualize recommendation system
            print("\n7. Generating recommendation visualizations...")
            
            # User-product interaction heatmap (sample)
            plt.figure(figsize=(12, 8))
            sample_matrix = user_product_matrix.iloc[:20, :20]  # Sample 20x20 for visualization
            sns.heatmap(sample_matrix, cmap='YlOrRd', cbar_kws={'label': 'Interaction Count'})
            plt.title('User-Product Interaction Matrix (Sample)')
            plt.xlabel('Products')
            plt.ylabel('Users')
            logger.report_matplotlib_figure(
                title="Recommendation System",
                series="User-Product Matrix",
                figure=plt.gcf()
            )
            plt.close()
            
            # Product similarity heatmap (sample)
            plt.figure(figsize=(10, 8))
            sample_similarity = product_similarity_df.iloc[:15, :15]  # Sample 15x15
            sns.heatmap(sample_similarity, annot=False, cmap='coolwarm', center=0)
            plt.title('Product Similarity Matrix (Sample)')
            plt.xlabel('Products')
            plt.ylabel('Products')
            logger.report_matplotlib_figure(
                title="Recommendation System",
                series="Product Similarity",
                figure=plt.gcf()
            )
            plt.close()
            
            # Interaction distribution
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            user_interactions.hist(bins=30, alpha=0.7, edgecolor='black')
            plt.title('User Interaction Distribution')
            plt.xlabel('Number of Interactions')
            plt.ylabel('Number of Users')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            product_interactions.hist(bins=30, alpha=0.7, edgecolor='black')
            plt.title('Product Interaction Distribution')
            plt.xlabel('Number of Interactions')
            plt.ylabel('Number of Products')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            logger.report_matplotlib_figure(
                title="Recommendation System",
                series="Interaction Distributions",
                figure=plt.gcf()
            )
            plt.close()
        
        # 8. Save all models and data
        print("\n" + "=" * 30)
        print("SAVING MODELS AND ARTIFACTS")
        print("=" * 30)
        
        # Save clustering model
        clustering_model_path = 'trained_clustering_model.pkl'
        joblib.dump(final_kmeans, clustering_model_path)
        print(f"Saved clustering model to {clustering_model_path}")
        
        # Save processed data with clusters
        clustered_data_path = 'data_with_clusters.csv'
        df.to_csv(clustered_data_path, index=False)
        print(f"Saved clustered data to {clustered_data_path}")
        
        if args['train_recommendation']:
            # Save recommendation models
            user_product_matrix_path = 'user_product_matrix.pkl'
            product_similarity_path = 'product_similarity_matrix.pkl'
            user_similarity_path = 'user_similarity_matrix.pkl'
            
            joblib.dump(user_product_matrix, user_product_matrix_path)
            joblib.dump(product_similarity_df, product_similarity_path)
            joblib.dump(user_similarity_df, user_similarity_path)
            
            print(f"Saved user-product matrix to {user_product_matrix_path}")
            print(f"Saved product similarity matrix to {product_similarity_path}")
            print(f"Saved user similarity matrix to {user_similarity_path}")
        
        # Upload artifacts to ClearML
        print("\nUploading artifacts to ClearML...")
        
        task.upload_artifact('clustering_model', clustering_model_path)
        task.upload_artifact('clustered_data', clustered_data_path)
        task.upload_artifact('clustering_results', clustering_results)
        
        if args['train_recommendation']:
            task.upload_artifact('user_product_matrix', user_product_matrix_path)
            task.upload_artifact('product_similarity_matrix', product_similarity_path)
            task.upload_artifact('user_similarity_matrix', user_similarity_path)
            task.upload_artifact('recommendation_metrics', recommendation_metrics)
        
        # Upload model metadata
        model_metadata = {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(df),
            'clustering_algorithm': args['clustering_algorithm'],
            'n_clusters': optimal_k,
            'clustering_metrics': {
                'silhouette_score': final_silhouette,
                'calinski_harabasz_score': final_calinski,
                'davies_bouldin_score': final_davies_bouldin,
                'inertia': final_inertia
            },
            'recommendation_trained': args['train_recommendation']
        }
        
        if args['train_recommendation']:
            model_metadata['recommendation_metrics'] = recommendation_metrics
        
        task.upload_artifact('model_metadata', model_metadata)
        
        # Log final metrics to ClearML
        logger.report_single_value("Clustering/Silhouette Score", final_silhouette)
        logger.report_single_value("Clustering/Calinski-Harabasz Score", final_calinski)
        logger.report_single_value("Clustering/Davies-Bouldin Score", final_davies_bouldin)
        logger.report_single_value("Clustering/Inertia", final_inertia)
        logger.report_single_value("Clustering/Number of Clusters", optimal_k)
        
        if args['train_recommendation']:
            logger.report_single_value("Recommendation/Total Interactions", recommendation_metrics['total_interactions'])
            logger.report_single_value("Recommendation/Unique Users", recommendation_metrics['unique_users'])
            logger.report_single_value("Recommendation/Unique Products", recommendation_metrics['unique_products'])
            logger.report_single_value("Recommendation/Matrix Sparsity", recommendation_metrics['matrix_sparsity'])
        
        print("\n" + "=" * 50)
        print("STEP 3 COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Clustering Model:")
        print(f"  Algorithm: {args['clustering_algorithm']}")
        print(f"  Number of clusters: {optimal_k}")
        print(f"  Silhouette score: {final_silhouette:.3f}")
        print(f"  Samples clustered: {len(df)}")
        
        if args['train_recommendation']:
            print(f"Recommendation Model:")
            print(f"  Users: {recommendation_metrics['unique_users']}")
            print(f"  Products: {recommendation_metrics['unique_products']}")
            print(f"  Interactions: {recommendation_metrics['total_interactions']}")
            print(f"  Matrix sparsity: {recommendation_metrics['matrix_sparsity']:.2%}")
        
        # Clean up temporary files
        temp_files = [clustering_model_path, clustered_data_path]
        if args['train_recommendation']:
            temp_files.extend([user_product_matrix_path, product_similarity_path, user_similarity_path])
        
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        
        return optimal_k, final_silhouette
        
    except Exception as e:
        print(f"Error in step 3: {str(e)}")
        raise e


if __name__ == '__main__':
    train_model()
