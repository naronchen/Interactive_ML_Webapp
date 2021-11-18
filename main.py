from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA #transform features to 2D

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt



def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


def add_param_ui(clf_name): #classfier name
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimator", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

def get_classifer(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["K"])
    elif clf_name == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                                    max_depth = params["max_depth"], 
                                    random_state=1234)

    return clf


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Mine Dataset"))
X,y = get_dataset(dataset_name)

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest")) #three different classifier
st.title(f"{dataset_name} dataset using {classifier_name}" )
st.write("Shape of dataset", X.shape, " Number of classes", len(np.unique(y)))

params = add_param_ui(classifier_name) #add ui, return params(from user input ) necessary for running the classifer
clf = get_classifer(classifier_name, params)

#Classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)


st.write(f"Accuracy = {acc}")


# Plot
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
fig.set_size_inches(17.5, 12.5)
plt.scatter(x1, x2, c=y, alpha = 0.8, cmap = "viridis")
plt.xlabel("Principle Component 1")
plt.ylabel("Principal Component 2")
plt.title("original data distribution")
plt.colorbar()
st.pyplot(fig)