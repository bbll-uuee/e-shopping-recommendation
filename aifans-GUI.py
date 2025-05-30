import streamlit as st
import pandas as pd
import joblib
import os

# ======== 1. Load Models and Data ========
@st.cache_resource
def load_models():
    user_matrix = joblib.load('smartphone_user_product_matrix.pkl')
    user_sim = joblib.load('smartphone_user_similarity.pkl')
    prod_sim = joblib.load('smartphone_product_similarity.pkl')
    info_df = pd.read_csv('product_info.csv')
    return user_matrix, user_sim, prod_sim, info_df

user_matrix, user_sim, prod_sim, product_info = load_models()

# ======== 2. Page Title ========
st.title("📦 E-Commerce Recommendation System - Smartphones & Accessories")

st.sidebar.header("🔍 Choose Recommendation Mode")
mode = st.sidebar.radio("Select a recommendation mode:", ["By User", "By Product"])

# ======== 3. User-Based Recommendation ========
if mode == "By User":
    st.subheader("👤 Recommend Products for a User")
    user_id = st.selectbox("Select a User ID", user_matrix.index.tolist())

    if st.button("Recommend for this User"):
        user_items = user_matrix.loc[user_id]
        owned = set(user_items[user_items > 0].index)

        # Find similar users
        similar_users = user_sim.loc[user_id].sort_values(ascending=False).drop(user_id).head(10).index
        rec_scores = {}

        for sim_user in similar_users:
            for item, qty in user_matrix.loc[sim_user].items():
                if item not in owned and qty > 0:
                    rec_scores[item] = rec_scores.get(item, 0) + qty

        top_items = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        rec_names = [i[0] for i in top_items]

        st.success("Top product recommendations for this user:")
        st.table(product_info[product_info['product_name'].isin(rec_names)][[
            'product_name', 'discounted_price', 'rating', 'rating_count'
        ]].rename(columns={
            'product_name': 'Product Name',
            'discounted_price': 'Price (₹)',
            'rating': 'Rating',
            'rating_count': 'Number of Ratings'
        }))

# ======== 4. Product-Based Recommendation ========
else:
    st.subheader("🔁 Recommend Similar Products")
    product_name = st.selectbox("Select a Product", prod_sim.columns.tolist())

    if st.button("Recommend Similar Products"):
        sim_scores = prod_sim.loc[product_name].sort_values(ascending=False).drop(product_name)
        top_similar = sim_scores.head(5).index.tolist()

        st.success("Top similar products:")
        st.table(product_info[product_info['product_name'].isin(top_similar)][[
            'product_name', 'discounted_price', 'rating', 'rating_count'
        ]].rename(columns={
            'product_name': 'Product Name',
            'discounted_price': 'Price (₹)',
            'rating': 'Rating',
            'rating_count': 'Number of Ratings'
        }))

# ======== 5. Footer ========
st.markdown("---")
st.markdown("This system recommends smartphone and accessory products based on user behavior and product similarity.")

