import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

encoded_data = pd.read_csv("data/preprocessed_mushrooms.csv")

@st.cache_resource
def feature_importance_plot():
    feat_names = encoded_data.columns[1:]
    rf = joblib.load("model/RF_feature_scoring_model.pkl") 
    feat_imps = rf.feature_importances_
    indices = np.argsort(feat_imps)
    plt.figure(figsize=(10,5))
    plt.barh([feat_names[i] for i in indices], [feat_imps[i] for i in indices], align='center')
    plt.title("Feature Importance Scores")
    st.pyplot(plt)
    st.caption("Horizontal Bar Chart displaying the features and their corresponding importance scores in descending order.")

@st.cache_resource
def heatmap_plot():
    plt.figure(figsize=(12.5,8))
    correlation = encoded_data.corr(method='pearson')
    sns.heatmap(correlation, annot=True, fmt='.2f', center=0.0)
    plt.tight_layout()
    st.pyplot(plt)
    st.caption("Heatmap to visualize the correlation between the mushroom class and all features, as wel as between all pairs of features.")

@st.cache_resource
def scree_plot():
    pca = PCA()
    pca.fit(encoded_data.drop(columns=['class']))
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(cumulative)+1), cumulative, marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.grid(True)
    st.pyplot(plt)
    st.caption("Scree Plot showing number of Principal Components needed to explain different levels of variance in the dataset.")


st.set_page_config(
    page_title="Feature Selection",
    page_icon="✅",
    layout="centered"
)

st.title("Feature Selection")

st.write("""To reduce the dimensionality of the dataset while maintaining classification performance, 
         three different feature selection techniques are used and their effectiveness compared:\n
- Feature Importances from a Random Forest Classifier
- Correlation with Target Class
- Principal Component Analysis (PCA)""")

st.subheader("Random Forest Feature Importances")
st.write("""A Random Forest classifier (with hyperparameter tuning) is trained on the full dataset,
         and `.feature_importances_` is used to rank all the features based on their contribution to
         classification. This model-based method captures non-linear relationships and interactions 
         between features, providing a strong indication of which variables are most informative.
""")
feature_importance_plot()


st.subheader("Correlation with Target Class")
st.write("""A correlation matrix is used to assess how relevant each feature is with respect to the 
         the target class. The absolute correlation values are then sorted in descending order to 
         identify the features most closely associated with class labels. While correlation is a simple 
         and fast technique, it only captures linear and individual relationships.
""")
heatmap_plot()


st.subheader("Principal Component Analysis (PCA)")
st.write("""PCA is applied to the feature set to reduce dimensionality by projecting the original features into a set of uncorrelated 
         principal components. These components are ranked based on the amount of variance they explain in the dataset. Although PCA 
         helps compress the data, it transforms the original features into abstract combinations, reducing interpretability.
""")
scree_plot()


st.subheader("Comparing the Techniques")
st.write("""To fairly compare the three methods, the same evaluation strategy is applied to each:\n
- A fresh Random Forest Classifier is trained on increasing subsets of the top-ranked features.
- Starting with the single top-most feature, and then adding one at a time, the performance of 
         the model is checked at each step.
- Finally, the method that results in the **highest accuracy** with **minimum number of features** is determined.\n
This brute-force performance check ensures that the features are selected, not by simply relying on scores or 
correlation values, but by actually testing how well the selected features support classification.
""")


st.subheader("Final Selection")
st.write("""Since, this data is very clean and synthetic it is possible to achieve 100% accuracy.
""")
st.markdown("###### Minimum number of features needed to reach 100% accuracy --")
col1, col2, col3 = st.columns(spec=3, gap="small")
with col1:
    st.metric(label="Random Forest", value=6)
with col2:
    st.metric(label="Correlation", value=12)
with col3:
    st.metric(label="PCA", value=14)

st.write("""So, the top 6 features obtained via the Random Forest feature importance scores are selected as the final feature 
         set. These features are used for all further modeling.""")
