import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Logistic Regression Learner", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1. Dataset Input", "2. Preprocessing & EDA", "3. Learning Module", "4. Training & Results"])

if page == "1. Dataset Input":
    st.header("📂 1. Dataset Input Module")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.write("### Dataset Preview", df.head())
        st.write("Shape:", df.shape)
        st.write("Data Types:", df.dtypes)

elif page == "2. Preprocessing & EDA":
    st.header("📊 2. Preprocessing & Exploratory Data Analysis")
    if 'df' in st.session_state:
        df = st.session_state['df']
        target_col = st.selectbox("Identify Target Variable", df.columns)
        
        st.subheader("EDA Visualization")
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.countplot(x=target_col, data=df, ax=ax1)
            ax1.set_title("Class Distribution")
            st.pyplot(fig1)
        with col2:
            fig2, ax2 = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
            ax2.set_title("Correlation Heatmap")
            st.pyplot(fig2)
        
        st.session_state['target'] = target_col
    else:
        st.warning("Please upload data first!")

elif page == "3. Learning Module":
    st.header("🧠 3. Logistic Regression Learning Module")
    st.write("### Why Sigmoid?")
    st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
    st.info("Logistic Regression converts linear output 'z' into a probability between 0 and 1 using the Sigmoid function.")
    
    z = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-z))
    fig, ax = plt.subplots()
    ax.plot(z, sigmoid)
    ax.set_title("Sigmoid Function")
    ax.set_xlabel("z")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

elif page == "4. Training & Results":
    st.header("📈 4. Training & Evaluation Metrics")
    if 'df' in st.session_state and 'target' in st.session_state:
        df = st.session_state['df']
        target = st.session_state['target']
        
        X = df.drop(target, axis=1).select_dtypes(include=[np.number])
        y = LabelEncoder().fit_transform(df[target])
        
        test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))
        with col2:
            st.write("### Classification Report")
            st.text(classification_report(y_test, y_pred))
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title("ROC Curve")
        st.pyplot(fig_roc)
    else:
        st.warning("Please complete data input and target selection steps!")