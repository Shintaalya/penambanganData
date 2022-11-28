import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

st.write(""" 
# Penambangan data
""")

st.write("=========================================================================")

tab1, tab2, tab3, tab4 = st.tabs(["Import Data", "Preprocessing", "Modelling", "Evalutions"])

with tab1:
    st.write("Import Data")
    data = pd.read_csv("https://raw.githubusercontent.com/Shintaalya/Datafile/main/online_classroom_data.csv")
    st.dataframe(data)

with tab2:
    data.head()
    
    # separate target 

    # values
    matrices_X = X.iloc[:,0:10].values

    # classes
    matrices_Y = X.iloc[:,10].values

    X_1 = X.iloc[:,0:10].values
    Y_1 = X.iloc[:, -1].values

    # X_train, X_test, y_train, y_test = train_test_split(matrices_X, matrices_Y, test_size = percent_amount_of_test_data, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X_1, Y_1, test_size = percent_amount_of_test_data, random_state=0)

    st.write("Menampilkan Y_1")
    st.write(Y_1)
    
    st.write("Menampilkan X_1")
    st.write(X_1)
    ### Dictionary to store model and its accuracy

    model_accuracy = OrderedDict()

    ### Dictionary to store model and its precision

    model_precision = OrderedDict()

    ### Dictionary to store model and its recall

    model_recall = OrderedDict()
