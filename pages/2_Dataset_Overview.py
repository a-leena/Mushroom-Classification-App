import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dataset Overview",
    page_icon="📋",
    layout="centered"
)

st.title("Dataset Overview")
st.header("Dataset Description")
st.markdown("""The dataset used, that was obtained from 
            [Kaggle](https://www.kaggle.com/datasets/uciml/mushroom-classification?resource=download), 
            contains 8124 instances of mushrooms with 22 physical characteristics and a binary target/class label 
            - _'p'_ for **_poisonous_** and _'e'_ for **_edible_**.""")

data = pd.read_csv("data/mushrooms.csv")

st.markdown("###### Number of instances of each class --")
st.metric(label="🍽️ Edible mushrooms", value=len(data[data['class']=='e']))
st.metric(label="☠️ Poisonous mushrooms", value=len(data[data['class']=='p']))

st.write("""Even though it's not a 50-50 collection, the dataset is not 
         particularly unbalanced. The table below show a glimpse of the dataset.""")
st.dataframe(data.head())
st.caption("Sample of records from the dataset.")

with st.expander("📃 Attribute Summary (Click to see what the each feature code stands for)"):
    attribute_data = {
        "cap-shape": "bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s",
        "cap-surface": "fibrous=f, grooves=g, scaly=y, smooth=s",
        "cap-color": "brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y",
        "bruises": "bruises=t, no bruises=f",
        "odor": "almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s",
        "gill-attachment": "attached=a, descending=d, free=f, notched=n",
        "gill-spacing": "close=c, crowded=w, distant=d",
        "gill-size": "broad=b, narrow=n",
        "gill-color": "black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y",
        "stalk-shape": "enlarging=e, tapering=t",
        "stalk-root": "bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?",
        "stalk-surface-above-ring": "fibrous=f, scaly=y, silky=k, smooth=s",
        "stalk-surface-below-ring": "fibrous=f, scaly=y, silky=k, smooth=s",
        "stalk-color-above-ring": "brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y",
        "stalk-color-below-ring": "brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y",
        "veil-type": "partial=p, universal=u",
        "veil-color": "brown=n, orange=o, white=w, yellow=y",
        "ring-number": "none=n, one=o, two=t",
        "ring-type": "cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z",
        "spore-print-color": "black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y",
        "population": "abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y",
        "habitat": "grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d"
    }
    st.caption("View in full screen.")
    attribute_df = pd.DataFrame(list(attribute_data.items()), columns=["Feature", "Encoded Values"])
    st.dataframe(attribute_df, height=250, hide_index=True)


st.header("Data Preprocessing")
st.write("The unique values for each column are observed using the code snippet below.")
st.code("""
        for col in data.columns:
            print(col,"-")
            print("Number of Unique values:",len(data[col].unique()))
            print("Unique values:",data[col].unique())
            print()
        """, language="python")

st.write("""A significant number of _NaN_ (missing) values were found for the column _'stalk-root'_, therefore this feature is eliminated. 
         Secondly, _'veil-type'_ had just one value in the entire dataset, while all the other features had 2 or more non-null categorical 
         values. Since it is a redundant column and will not be helpful in classification, this feature is also removed.
         """)

st.code("""
        le = preprocessing.LabelEncoder()
        for col in data.columns:
            data[col] = le.fit_transform(data[col])
        data.head()
""", language="python")
st.write("""
         In order to proceed, these categorical features are first converted to numerical form. For this 
         `LabelEncoder()` from `sklearn.preprocessing` module is used (shown in the code above).
         This ensures that the values are numerically encoded based on the lexical order of unique values. 
         Thus, the target column _'class'_ has its values changed from _'e'_ and _'p'_ to _0_ and _1_, and so on.
         The table below shows a preview of the dataset after encoding.
""")

encoded_data = pd.read_csv("data/preprocessed_mushrooms.csv")
st.dataframe(encoded_data.head())
st.caption("Sample of records from the preprocessed dataset.")
