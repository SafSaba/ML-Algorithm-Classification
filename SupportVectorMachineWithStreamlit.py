import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="SVM Breast Cancer Classifier", layout="centered")
st.title("SVM Model for Breast Cancer Classification")


# 1. Load Data
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 2. Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train SVM Model
C_param = st.sidebar.slider("Select SVM Regularization Parameter (C)", 0.01, 10.0, 1.0)
svm_model = SVC(kernel='linear', C=C_param, random_state=42)  # Using a linear boundary
svm_model.fit(X_train, y_train)

# 5. Make Predictions
predictions = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# --- Display Results in Streamlit ---

# Display Accuracy
st.subheader("Model Accuracy")
st.write(f"The accuracy of the SVM model is: **{accuracy:.2%}**")

st.metric(label="SVM Model Accuracy", value=f"{accuracy:.2f}")


# Display Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, predictions)

# Create a matplotlib figure to plot the heatmap
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names,
            ax=ax)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVM Confusion Matrix')

# Display the plot in Streamlit
st.pyplot(fig)

# Display the raw data
if st.checkbox("Show raw data"):
    st.subheader("Breast Cancer Dataset")
    st.write(cancer.data)