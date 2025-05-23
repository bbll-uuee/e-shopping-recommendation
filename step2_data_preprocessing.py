import os
import pandas as pd
import numpy as np
from clearml import Task, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def preprocess_data():
    """
    Step 2: Data Preprocessing
    This function handles data cleaning, feature engineering, and preprocessing
    """
    
    # Initialize ClearML Task
    task = Task.init(
        project_name="examples",
        task_name="Pipeline step 2 process dataset",
        task_type=Task.TaskTypes.data_processing
    )
    
    # For remote execution - this line is key
    task.execute_remotely()
    
    # Get parameters from task (can be overridden from pipeline)
    args = {
        'dataset_task_id': '0fad9317e21c488da29a288ad50ec2fa',  # Will be overridden by pipeline
        'test_size': 0.25,
        'random_state': 42,
        'remove_outliers': True,
        'outlier_threshold': 3.0,  # Standard deviations for outlier detection
        'feature_scaling': True,
        'handle_missing_values': True
    }
    
    # Connect parameters to task
    task.connect(args)
    
    print("=" * 50)
    print("STEP 2: Data Preprocessing")
    print("=" * 50)
    
    try:
        # Get dataset from previous step
        if args['dataset_task_id']:
            print(f"Loading dataset from task: {args['dataset_task_id']}")
            dataset_task = Task.get_task(task_id=args['dataset_task_id'])
            dataset_id = dataset_task.artifacts['dataset_id'].get()
            dataset = Dataset.get(dataset_id=dataset_id)
            dataset_path = dataset.get_local_copy()
            
            # Load the full dataset
            full_dataset_file = os.path.join(dataset_path, 'full_dataset.csv')
            df = pd.read_csv(full_dataset_file)
            print(f"Loaded dataset with {len(df)} samples")
        else:
            raise ValueError("dataset_task_id not provided")
        
        print(f"Original dataset shape: {df.shape}")
        print("\nOriginal dataset info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Data Preprocessing Steps
        print("\n" + "=" * 30)
        print("DATA CLEANING & PREPROCESSING")
        print("=" * 30)
        
        # 1. Handle missing values
        if args['handle_missing_values']:
            print("\n1. Handling missing values...")
            print("Missing values before processing:")
            missing_before = df.isnull().sum()
            print(missing_before[missing_before > 0])
            
            # Fill missing values using mean for numerical columns
            numerical_columns = ['discounted_price', 'actual_price', 'rating', 'rating_count', 'Sales', 'Quantity', 'Profit']
            
            for col in numerical_columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        mean_value = df[col].mean()
                        df[col] = df[col].fillna(mean_value)
                        print(f"  Filled {missing_count} missing values in '{col}' with mean: {mean_value:.2f}")
            
            # Fill missing values for categorical columns
            categorical_columns = ['product_name']
            for col in categorical_columns:
                if col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown'
                        df[col] = df[col].fillna(mode_value)
                        print(f"  Filled {missing_count} missing values in '{col}' with mode: {mode_value}")
            
            print("Missing values after processing:")
            missing_after = df.isnull().sum()
            print(missing_after[missing_after > 0])
        
        # 2. Data type conversion and cleaning
        print("\n2. Data type conversion and cleaning...")
        
        # Ensure numerical columns are properly typed
        for col in ['discounted_price', 'actual_price', 'rating', 'rating_count', 'Sales', 'Quantity', 'Profit']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Standardize product names
        if 'product_name' in df.columns:
            df['product_name'] = df['product_name'].str.strip().str.lower()
            print(f"  Standardized product names: {df['product_name'].nunique()} unique products")
        
        # 3. Handle negative or unreasonable values
        print("\n3. Handling negative or unreasonable values...")
        
        # Check for negative values in columns that should be positive
        positive_columns = ['discounted_price', 'actual_price', 'Sales', 'Quantity', 'Profit']
        for col in positive_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    print(f"  Found {negative_count} negative values in '{col}', setting to 0")
                    df[col] = df[col].clip(lower=0)
        
        # Ensure rating is in valid range (1-5)
        if 'rating' in df.columns:
            invalid_ratings = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
            if invalid_ratings > 0:
                print(f"  Found {invalid_ratings} invalid ratings, clipping to 1-5 range")
                df['rating'] = df['rating'].clip(lower=1, upper=5)
        
        # Ensure discounted_price <= actual_price
        if 'discounted_price' in df.columns and 'actual_price' in df.columns:
            inconsistent_prices = (df['discounted_price'] > df['actual_price']).sum()
            if inconsistent_prices > 0:
                print(f"  Found {inconsistent_prices} cases where discounted_price > actual_price")
                df['actual_price'] = np.maximum(df['actual_price'], df['discounted_price'])
        
        # 4. Feature Engineering
        print("\n4. Feature engineering...")
        
        # Create new features
        if 'discounted_price' in df.columns and 'actual_price' in df.columns:
            df['discount_amount'] = df['actual_price'] - df['discounted_price']
            df['discount_percentage'] = (df['discount_amount'] / df['actual_price']) * 100
            print("  Created discount_amount and discount_percentage features")
        
        if 'Sales' in df.columns and 'Quantity' in df.columns:
            df['price_per_unit'] = df['Sales'] / (df['Quantity'] + 1e-6)  # Add small value to avoid division by zero
            print("  Created price_per_unit feature")
        
        if 'Profit' in df.columns and 'Sales' in df.columns:
            df['profit_margin'] = (df['Profit'] / (df['Sales'] + 1e-6)) * 100
            print("  Created profit_margin feature")
        
        # Create rating categories
        if 'rating' in df.columns:
            df['rating_category'] = pd.cut(df['rating'], 
                                         bins=[0, 2, 3, 4, 5], 
                                         labels=['poor', 'fair', 'good', 'excellent'])
            print("  Created rating_category feature")
        
        # 5. Outlier detection and removal
        if args['remove_outliers']:
            print(f"\n5. Outlier detection and removal (threshold: {args['outlier_threshold']} std)...")
            
            numerical_features = ['discounted_price', 'actual_price', 'rating', 'rating_count', 
                                'Sales', 'Quantity', 'Profit', 'discount_amount', 'price_per_unit', 'profit_margin']
            
            original_size = len(df)
            
            for col in numerical_features:
                if col in df.columns:
                    # Calculate z-scores
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outliers = z_scores > args['outlier_threshold']
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        print(f"  Found {outlier_count} outliers in '{col}'")
                        # Remove outliers
                        df = df[~outliers]
            
            removed_samples = original_size - len(df)
            print(f"  Removed {removed_samples} samples due to outliers")
            print(f"  Dataset size after outlier removal: {len(df)}")
        
        # 6. Prepare features for modeling
        print("\n6. Preparing features for modeling...")
        
        # Define feature columns for clustering
        clustering_features = ['discounted_price', 'actual_price', 'rating', 'rating_count', 'Profit']
        
        # Define feature columns for recommendation
        recommendation_features = ['user_id', 'product_name', 'Quantity', 'rating']
        
        # Check if all required features exist
        missing_clustering_features = [f for f in clustering_features if f not in df.columns]
        missing_recommendation_features = [f for f in recommendation_features if f not in df.columns]
        
        if missing_clustering_features:
            print(f"  Warning: Missing clustering features: {missing_clustering_features}")
        
        if missing_recommendation_features:
            print(f"  Warning: Missing recommendation features: {missing_recommendation_features}")
        
        # Create feature matrix for clustering
        clustering_data = df[clustering_features].copy()
        
        # 7. Feature scaling
        if args['feature_scaling']:
            print("\n7. Feature scaling...")
            
            scaler = StandardScaler()
            clustering_data_scaled = scaler.fit_transform(clustering_data)
            clustering_data_scaled_df = pd.DataFrame(clustering_data_scaled, 
                                                   columns=clustering_features,
                                                   index=clustering_data.index)
            
            print("  Applied StandardScaler to clustering features")
            
            # Save scaler for later use
            scaler_path = 'standard_scaler.pkl'
            joblib.dump(scaler, scaler_path)
            print(f"  Saved scaler to {scaler_path}")
        
        # 8. Create train-test split
        print(f"\n8. Creating train-test split (test_size: {args['test_size']})...")
        
        train_df, test_df = train_test_split(
            df, 
            test_size=args['test_size'],
            random_state=args['random_state'],
            stratify=None
        )
        
        print(f"  Training set size: {len(train_df)}")
        print(f"  Test set size: {len(test_df)}")
        
        # 9. Data validation and quality checks
        print("\n9. Data validation and quality checks...")
        
        # Check data distributions
        print("  Data distribution summary:")
        print(df[clustering_features].describe())
        
        # Check for any remaining issues
        print("  Final data quality check:")
        print(f"    Total samples: {len(df)}")
        print(f"    Total features: {len(df.columns)}")
        print(f"    Missing values: {df.isnull().sum().sum()}")
        print(f"    Duplicate rows: {df.duplicated().sum()}")
        
        # 10. Save processed data
        print("\n10. Saving processed data...")
        
        # Save datasets
        processed_train_file = 'processed_train_data.csv'
        processed_test_file = 'processed_test_data.csv'
        processed_full_file = 'processed_full_data.csv'
        clustering_features_file = 'clustering_features.csv'
        
        train_df.to_csv(processed_train_file, index=False)
        test_df.to_csv(processed_test_file, index=False)
        df.to_csv(processed_full_file, index=False)
        clustering_data.to_csv(clustering_features_file, index=False)
        
        print(f"  Saved processed datasets:")
        print(f"    - {processed_train_file}")
        print(f"    - {processed_test_file}")
        print(f"    - {processed_full_file}")
        print(f"    - {clustering_features_file}")
        
        # Create and upload artifacts
        print("\n11. Creating ClearML artifacts...")
        
        # Upload processed data as artifacts
        task.upload_artifact('processed_train_data', processed_train_file)
        task.upload_artifact('processed_test_data', processed_test_file)
        task.upload_artifact('processed_full_data', processed_full_file)
        task.upload_artifact('clustering_features', clustering_features_file)
        
        if args['feature_scaling']:
            task.upload_artifact('standard_scaler', scaler_path)
        
        # Upload metadata
        metadata = {
            'original_samples': len(df),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'features': list(df.columns),
            'clustering_features': clustering_features,
            'recommendation_features': recommendation_features,
            'outliers_removed': removed_samples if args['remove_outliers'] else 0,
            'feature_scaling_applied': args['feature_scaling']
        }
        
        task.upload_artifact('preprocessing_metadata', metadata)
        
        # 12. Generate visualization reports
        print("\n12. Generating visualization reports...")
        
        logger = task.get_logger()
        
        try:
            # Data distribution plots
            plt.figure(figsize=(15, 10))
            
            # Plot distributions of key features
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Rating distribution
            axes[0, 0].hist(df['rating'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Rating Distribution')
            axes[0, 0].set_xlabel('Rating')
            axes[0, 0].set_ylabel('Frequency')
            
            # Price distributions
            axes[0, 1].hist(df['discounted_price'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Discounted Price Distribution')
            axes[0, 1].set_xlabel('Price')
            axes[0, 1].set_ylabel('Frequency')
            
            axes[0, 2].hist(df['actual_price'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Actual Price Distribution')
            axes[0, 2].set_xlabel('Price')
            axes[0, 2].set_ylabel('Frequency')
            
            # Sales, Quantity, Profit distributions
            axes[1, 0].hist(df['Sales'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Sales Distribution')
            axes[1, 0].set_xlabel('Sales')
            axes[1, 0].set_ylabel('Frequency')
            
            axes[1, 1].hist(df['Quantity'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Quantity Distribution')
            axes[1, 1].set_xlabel('Quantity')
            axes[1, 1].set_ylabel('Frequency')
            
            axes[1, 2].hist(df['Profit'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 2].set_title('Profit Distribution')
            axes[1, 2].set_xlabel('Profit')
            axes[1, 2].set_ylabel('Frequency')
            
            plt.tight_layout()
            logger.report_matplotlib_figure(
                title="Feature Distributions",
                series="After Preprocessing",
                figure=plt.gcf()
            )
            plt.close()
            
            # Correlation heatmap
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[clustering_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Feature Correlation Matrix')
            logger.report_matplotlib_figure(
                title="Feature Analysis",
                series="Correlation Matrix",
                figure=plt.gcf()
            )
            plt.close()
            
            # New features distribution
            if 'discount_percentage' in df.columns:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.hist(df['discount_percentage'], bins=30, alpha=0.7, edgecolor='black')
                plt.title('Discount Percentage Distribution')
                plt.xlabel('Discount %')
                plt.ylabel('Frequency')
                
                plt.subplot(1, 3, 2)
                plt.hist(df['price_per_unit'], bins=30, alpha=0.7, edgecolor='black')
                plt.title('Price per Unit Distribution')
                plt.xlabel('Price per Unit')
                plt.ylabel('Frequency')
                
                plt.subplot(1, 3, 3)
                plt.hist(df['profit_margin'], bins=30, alpha=0.7, edgecolor='black')
                plt.title('Profit Margin Distribution')
                plt.xlabel('Profit Margin %')
                plt.ylabel('Frequency')
                
                plt.tight_layout()
                logger.report_matplotlib_figure(
                    title="Engineered Features",
                    series="Distribution",
                    figure=plt.gcf()
                )
                plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate all visualizations: {e}")
        
        # Log summary statistics
        logger.report_table(
            title="Preprocessing Summary",
            series="Statistics",
            table_plot=df[clustering_features].describe()
        )
        
        print("\n" + "=" * 50)
        print("STEP 2 COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Processed dataset samples: {len(df)}")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"Clustering features: {len(clustering_features)}")
        print(f"Outliers removed: {removed_samples if args['remove_outliers'] else 0}")
        
        # Clean up temporary files
        temp_files = [processed_train_file, processed_test_file, processed_full_file, 
                     clustering_features_file]
        if args['feature_scaling']:
            temp_files.append(scaler_path)
            
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        
        return len(df)
        
    except Exception as e:
        print(f"Error in step 2: {str(e)}")
        raise e


if __name__ == '__main__':
    preprocess_data()
