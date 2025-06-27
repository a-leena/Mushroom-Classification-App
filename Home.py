import streamlit as st


st.set_page_config(
    page_title="Mushroom Classification", 
    page_icon="🍄",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.sidebar.markdown("🔗[Github Repository](https://github.com/a-leena/Mushroom-Classification-App/tree/main/notebooks)")

st.title("Mushroom Classification Project")
st.header("Project Overview")
st.write("The objective of this project is to classify mushrooms as **_edible_** or **_poisonous_** based on their physical features.")
st.markdown("""To carry out this task in Python some essential libraries used are _numpy_ and _pandas_ 
            for data handling and preprocessing, _sklearn_ (scikit-learn) for machine learning, and _matplotlib_ and 
            _seaborn_ for data visualization.""")
st.divider()
st.header("🗺️ What You Can Explore")
col1, col2 = st.columns(spec=2, gap="large")

with col1:
    st.subheader("Classifier")
    st.write("Choose the mushroom's physical traits and predict if it's edible or poisonous.")

with col2:
    st.subheader("Dataset Overview")
    st.write("Understand the structure, attributes, and preprocessing of the mushroom dataset.")
    
col3, col4 = st.columns(spec=2, gap="large")
with col3:
    st.subheader("Feature Selection")
    st.write("See how the dataset was reduced from 22 features to the 6 most predictive ones.")


with col4:
    st.subheader("Model Comparisons")
    st.write("Explore how various models performed and why the final one was chosen.")

