# e-shopping-recommendation
📦 E-commerce Recommendation System

This project builds a cluster-based recommendation system for e-commerce platforms using ClearML, scikit-learn, and Streamlit. It supports full automation of the ML lifecycle — from data preprocessing, model training, hyperparameter optimization, to deployment-ready recommendation components.

🧠 Features
📊 Clustering with KMeans / MiniBatchKMeans
Discover customer segments with unsupervised learning.
🎯 ClearML Hyperparameter Optimization (HPO)
Automatically find optimal parameters for clustering & recommendations.
🤝 Collaborative Filtering with Cosine Similarity
Recommend similar products or generate user-specific suggestions.
📈 Interactive Dashboards
Visualize clustering metrics and recommendation matrices directly in ClearML.
🖥️ Streamlit GUI
A clean, simple web interface to test recommendations by user or product.

🚀 Pipeline Structure
step1_create_dataset
        ↓
step2_preprocess
        ↓
step3_initial_train
        ↓
step4_hpo
        ↓
step5_final_model


📷 Sample UI (Streamlit)
User-Based Recommendations:
Product-Based Recommendations:

📦 Getting Started
1. Install dependencies
   pip install -r requirements.txt
2. Run pipeline (via ClearML)
   python pipeline_from_tasks.py
3. Launch Streamlit GUI
   streamlit run app.py
