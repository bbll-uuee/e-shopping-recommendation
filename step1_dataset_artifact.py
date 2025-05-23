import os
import pandas as pd
import numpy as np
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split


def create_dataset_artifact():
    """
    Step 1: Create and manage dataset artifacts
    This function handles data loading, basic validation, and creates ClearML dataset artifacts
    """
    
    # Initialize ClearML Task
    task = Task.init(
        project_name="examples",
        task_name="Pipeline step 1 dataset artifact",
        task_type=Task.TaskTypes.data_processing
    )
    
    # For remote execution - this line is key
    task.execute_remotely()
    
    # Get parameters from task (can be overridden from pipeline)
    args = {
        'dataset_name': 'amazon_ecommerce_data',
        'dataset_project': 'examples',
        'train_test_split_ratio': 0.8,
        'random_state': 42,
        'data_validation': True
    }
    
    # Connect parameters to task
    task.connect(args)
    
    print("=" * 50)
    print("STEP 1: Dataset Artifact Creation")
    print("=" * 50)
    
    try:
        # Create dataset - In real scenario, you would load from your actual data source
        # Here we'll create a sample dataset similar to your Amazon data structure
        print("Creating sample Amazon e-commerce dataset...")
        
        # Generate sample data that matches your model structure
        np.random.seed(args['random_state'])
        n_samples = 10000
        n_users = 1000
        n_products = 500
        
        # Create sample data
        data = {
            'user_id': np.random.randint(1, n_users + 1, n_samples),
            'product_name': [f'product_{np.random.randint(1, n_products + 1)}' for _ in range(n_samples)],
            'discounted_price': np.random.uniform(10, 500, n_samples),
            'actual_price': np.random.uniform(15, 600, n_samples),
            'rating': np.random.uniform(1, 5, n_samples),
            'rating_count': np.random.randint(1, 1000, n_samples),
            'Sales': np.random.uniform(100, 10000, n_samples),
            'Quantity': np.random.randint(1, 100, n_samples),
            'Profit': np.random.uniform(10, 200, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Ensure actual_price >= discounted_price
        df['actual_price'] = np.maximum(df['actual_price'], df['discounted_price'])
        
        print(f"Generated dataset with {len(df)} samples")
        print(f"Dataset shape: {df.shape}")
        print("\nDataset info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Basic data validation
        if args['data_validation']:
            print("\nPerforming data validation...")
            
            # Check for missing values
            missing_values = df.isnull().sum()
            print(f"Missing values per column:\n{missing_values}")
            
            # Check for negative values in key columns
            negative_checks = {
                'discounted_price': (df['discounted_price'] < 0).sum(),
                'actual_price': (df['actual_price'] < 0).sum(),
                'rating': (df['rating'] < 0).sum(),
                'Sales': (df['Sales'] < 0).sum(),
                'Quantity': (df['Quantity'] < 0).sum(),
                'Profit': (df['Profit'] < 0).sum()
            }
            
            print("Negative values check:")
            for col, count in negative_checks.items():
                print(f"  {col}: {count} negative values")
            
            # Check rating range
            rating_out_of_range = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
            print(f"Ratings out of 1-5 range: {rating_out_of_range}")
            
            # Price consistency check
            price_inconsistent = (df['discounted_price'] > df['actual_price']).sum()
            print(f"Discounted price > actual price: {price_inconsistent}")
        
        # Create train-test split
        print(f"\nSplitting data with ratio {args['train_test_split_ratio']}")
        train_df, test_df = train_test_split(
            df, 
            test_size=1 - args['train_test_split_ratio'],
            random_state=args['random_state'],
            stratify=None  # No stratification for this dataset
        )
        
        print(f"Training set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        
        # Save datasets to temporary files
        train_file = 'train_data.csv'
        test_file = 'test_data.csv'
        full_dataset_file = 'full_dataset.csv'
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        df.to_csv(full_dataset_file, index=False)
        
        print(f"\nSaved datasets to:")
        print(f"  - {train_file}")
        print(f"  - {test_file}")
        print(f"  - {full_dataset_file}")
        
        # Create ClearML Dataset
        print(f"\nCreating ClearML dataset: {args['dataset_name']}")
        
        dataset = Dataset.create(
            dataset_name=args['dataset_name'],
            dataset_project=args['dataset_project'],
            parent_datasets=None
        )
        
        # Add files to dataset
        dataset.add_files(path=train_file)
        dataset.add_files(path=test_file)
        dataset.add_files(path=full_dataset_file)
        
        # Add metadata
        dataset.get_logger().report_text(
            f"Dataset created with {len(df)} total samples\n"
            f"Training samples: {len(train_df)}\n"
            f"Test samples: {len(test_df)}\n"
            f"Features: {list(df.columns)}"
        )
        
        # Upload and finalize dataset
        dataset.upload()
        dataset.finalize()
        
        dataset_id = dataset.id
        print(f"Dataset created successfully with ID: {dataset_id}")
        
        # Log dataset statistics to ClearML
        logger = task.get_logger()
        
        # Log basic statistics
        logger.report_table(
            title="Dataset Statistics",
            series="Basic Info",
            table_plot=df.describe()
        )
        
        # Log data distribution plots if possible
        try:
            import matplotlib.pyplot as plt
            
            # Rating distribution
            plt.figure(figsize=(10, 6))
            plt.hist(df['rating'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Rating Distribution')
            plt.xlabel('Rating')
            plt.ylabel('Frequency')
            logger.report_matplotlib_figure(
                title="Data Distribution",
                series="Rating Distribution",
                figure=plt.gcf()
            )
            plt.close()
            
            # Price distribution
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.hist(df['discounted_price'], bins=30, alpha=0.7, edgecolor='black')
            plt.title('Discounted Price Distribution')
            plt.xlabel('Price')
            plt.ylabel('Frequency')
            
            plt.subplot(1, 2, 2)
            plt.hist(df['actual_price'], bins=30, alpha=0.7, edgecolor='black')
            plt.title('Actual Price Distribution')
            plt.xlabel('Price')
            plt.ylabel('Frequency')
            
            plt.tight_layout()
            logger.report_matplotlib_figure(
                title="Data Distribution",
                series="Price Distribution",
                figure=plt.gcf()
            )
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for plotting")
        
        # Store dataset ID for next steps
        task.upload_artifact('dataset_id', dataset_id)
        task.upload_artifact('train_samples', len(train_df))
        task.upload_artifact('test_samples', len(test_df))
        task.upload_artifact('feature_columns', list(df.columns))
        
        print("\n" + "=" * 50)
        print("STEP 1 COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print(f"Dataset artifact created: {dataset_id}")
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"Total features: {len(df.columns)}")
        
        # Clean up temporary files
        for file in [train_file, test_file, full_dataset_file]:
            if os.path.exists(file):
                os.remove(file)
        
        return dataset_id
        
    except Exception as e:
        print(f"Error in step 1: {str(e)}")
        raise e


if __name__ == '__main__':
    create_dataset_artifact()
