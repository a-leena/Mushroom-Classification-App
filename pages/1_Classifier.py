import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    dt_model = joblib.load('model/decision_tree.pkl')
    encoders = joblib.load('model/encoders.pkl')
    return dt_model, encoders

feature_options = {
    "odor": {
        "almond": "a", "anise": "l", "creosote": "c", "fishy": "y", "foul": "f",
        "musty": "m", "pungent": "p", "spicy": "s", "none": "n"
    },
    "gill-size": {
        "broad": "b", "narrow": "n"
    },
    "gill-color": {
        "black": "k", "brown": "n", "buff": "b", "chocolate": "h", "gray": "g",
        "green": "r", "orange": "o", "pink": "p", "purple": "u", "red": "e",
        "white": "w", "yellow": "y"
    },
    "spore-print-color": {
        "black": "k", "brown": "n", "buff": "b", "chocolate": "h", "green": "r",
        "orange": "o", "purple": "u", "white": "w", "yellow": "y"
    },
    "population": {
        "abundant": "a", "clustered": "c", "numerous": "n", "scattered": "s",
        "several": "v", "solitary": "y"
    },
    "stalk-surface-above-ring": {
        "fibrous": "f", "scaly": "y", "silky": "k", "smooth": "s"
    }
}

st.set_page_config(
    page_title="Classifier",
    page_icon="🤖",
    layout="centered"
)

st.title("Mushroom Classifier")
st.write("Select values for the features below and click **Predict** to see whether the mushroom is edible or poisonous.")

with st.form("prediction_form"):
    odor = st.selectbox("Odor", list(feature_options["odor"].keys()))
    gill_size = st.selectbox("Gill-Size", list(feature_options["gill-size"].keys()))
    gill_color = st.selectbox("Gill-Color", list(feature_options["gill-color"].keys()))
    spore_print_color = st.selectbox("Spore-Print Color", list(feature_options["spore-print-color"].keys()))
    population = st.selectbox("Population", list(feature_options["population"].keys()))
    stalk_surface_above_ring = st.selectbox("Stalk-Surface Above Ring", list(feature_options["stalk-surface-above-ring"].keys()))

    submit = st.form_submit_button("Predict")


if submit:
    dt_model, encoders = load_models()
    input_labels = {
        "odor": feature_options["odor"][odor],
        "gill-size": feature_options["gill-size"][gill_size],
        "gill-color": feature_options["gill-color"][gill_color],
        "spore-print-color": feature_options["spore-print-color"][spore_print_color],
        "population": feature_options["population"][population],
        "stalk-surface-above-ring": feature_options["stalk-surface-above-ring"][stalk_surface_above_ring]
    }
    input_encoded = {
        feature: int(encoders[feature].transform([code])[0])
        for feature, code in input_labels.items()
    }
    input_df = pd.DataFrame(data=[input_encoded])
    prediction = dt_model.predict(input_df)[0]
    result = "🍽️Edible" if prediction==0 else "☠️Poisonous"
    st.success(f"##### This mushroom is predicted to be **{result}** !")
