import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_initial_graph(dataset):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
    
    df = pd.DataFrame({'Feature1': X[:, 0], 'Feature2': X[:, 1], 'Target': y})
    return X, y, df

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.1)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.1)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

st.sidebar.markdown("# Logistic Regression Classifier")

dataset = st.sidebar.selectbox('Select Dataset', ('Binary', 'Multiclass'))
penalty = st.sidebar.selectbox('Regularization', ('l2', 'l1', 'elasticnet', 'none'))
c_input = float(st.sidebar.number_input('C', value=1.0))
solver = st.sidebar.selectbox('Solver', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
max_iter = int(st.sidebar.number_input('Max Iterations', value=100))
multi_class = st.sidebar.selectbox('Multi Class', ('auto', 'ovr', 'multinomial'))
l1_ratio = float(st.sidebar.number_input('l1 Ratio', value=0.5))

X, y, df = load_initial_graph(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

st.subheader("Initial Data Distribution")
fig = sns.scatterplot(x=df['Feature1'], y=df['Feature2'], hue=df['Target'], palette='rainbow')
st.pyplot(fig.figure)

if st.sidebar.button('Run Algorithm'):
    clf = LogisticRegression(
        penalty=penalty,
        C=c_input,
        solver=solver,
        max_iter=max_iter,
        multi_class=multi_class,
        l1_ratio=l1_ratio if penalty == "elasticnet" else None
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    XX, YY, input_array = draw_meshgrid(X)
    labels = clf.predict(input_array)
    mesh_df = pd.DataFrame({'Feature1': input_array[:, 0], 'Feature2': input_array[:, 1], 'Target': labels})
    
    st.subheader("Decision Boundary")
    fig = sns.scatterplot(data=mesh_df, x='Feature1', y='Feature2', hue='Target', alpha=0.3, palette='rainbow')
    sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Target', edgecolor='black', palette='rainbow')
    st.pyplot(fig.figure)
    
    st.subheader(f"Accuracy for Logistic Regression: {round(accuracy_score(y_test, y_pred), 2)}")
