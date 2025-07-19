# KNeighborsClassifierWithStreamlit.py
# This script implements a Streamlit app for classifying Iris species using K-Nearest Neighbors
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# --- Page Title ---
st.title("Iris Species Classification with KNN")
st.write("This app uses the K-Nearest Neighbors algorithm to predict the species of an Iris flower.")

# --- Load Data ---
# No caching needed for this small dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
target_names = iris.target_names

# --- Train Model ---
# We train the model on the entire dataset for this interactive app
model = KNeighborsClassifier(n_neighbors=5)
model.fit(df.iloc[:, :-1], df['species'])

# --- Sidebar for User Input ---
st.sidebar.title("Input Features")
st.sidebar.header("Adjust the sliders to input flower measurements:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

# --- Prediction ---
# Create a list from the slider inputs to feed the model
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Get the prediction from the model
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

# --- Display Prediction ---
st.write("---")
st.header("Prediction Result")
st.write(f"Based on the input, the predicted species is:")
st.markdown(f"## **{predicted_species.capitalize()}**")

# Display an image corresponding to the prediction
if predicted_species == 'setosa':
    st.image('https://placehold.co/400x300/E8D4F2/5D3A7A?text=Setosa', caption='Iris Setosa')
elif predicted_species == 'versicolor':
    st.image('https://placehold.co/400x300/D4E8F2/3A5D7A?text=Versicolor', caption='Iris Versicolor')
else:
    st.image('https://placehold.co/400x300/D4F2E8/3A7A5D?text=Virginica', caption='Iris Virginica')

