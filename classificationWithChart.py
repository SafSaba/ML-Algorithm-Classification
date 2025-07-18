import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Caching ---
# Use Streamlit's caching to load data only once, improving performance.
@st.cache_data
def load_data():
    """Loads the Iris dataset and returns a DataFrame and target names."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species_id'] = iris.target
    # Map species IDs to names for readability
    df['species_name'] = df['species_id'].apply(lambda x: iris.target_names[x])
    return df, iris.target_names

# --- 2. Model Training ---
# This function also gets cached so the model is trained only once.
@st.cache_resource
def train_model(df):
    """Trains a RandomForestClassifier on the provided DataFrame."""
    X = df.iloc[:, :4]  # Features (first 4 columns)
    y = df['species_id'] # Target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- App Initialization ---
# Load data and train the model
df, target_names = load_data()
model = train_model(df)

# --- 3. Page Configuration and Sidebar ---
st.set_page_config(page_title="Iris Species Predictor", layout="wide")
st.sidebar.title("Input Features")
st.sidebar.header("Adjust the sliders to match your flower's measurements.")

# Create sliders in the sidebar for user input.
# The values are dynamically set based on the min/max values in the dataset.
sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), value=5.4)
sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), value=3.4)
petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), value=1.3)
petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), value=0.2)

# --- 4. Main Panel Display ---
st.title("Iris Flower Species Predictor 🌸")
st.write("This app uses a Random Forest model to predict the species of an Iris flower based on its measurements. Adjust the sliders on the left and see the prediction change in real-time!")

# Organize user input into a DataFrame for display
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

st.header("Your Input")
st.dataframe(input_data)

# --- 5. Prediction and Probabilities ---
# Get prediction and prediction probabilities
prediction_id = model.predict(input_data)[0]
prediction_name = target_names[prediction_id]
prediction_proba = model.predict_proba(input_data)

# Display the prediction results
st.header("Prediction Result")
col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Species", prediction_name.capitalize())
with col2:
    st.metric("Confidence", f"{prediction_proba.max()*100:.2f}%")

# Display the probabilities for each class
st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.dataframe(proba_df)


# --- 6. Data Visualization ---
st.header("Visualize Your Input")
st.write("See how your input (the red X) compares to the original dataset.")

# Create two columns for two different plots
fig1, ax1 = plt.subplots()
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue='species_name', ax=ax1, palette="viridis")
ax1.scatter(sepal_length, sepal_width, marker='X', color='red', s=100, label='Your Input')
ax1.set_title("Sepal Length vs. Sepal Width")
ax1.legend()

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species_name', ax=ax2, palette="viridis")
ax2.scatter(petal_length, petal_width, marker='X', color='red', s=100, label='Your Input')
ax2.set_title("Petal Length vs. Petal Width")
ax2.legend()

# Display plots in Streamlit
col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)


# --- 7. Optional: Show Raw Data ---
with st.expander("See the full dataset used for training"):
    st.dataframe(df)

