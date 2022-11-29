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
# Kelayakan Makan Jamur
""")

st.write("=========================================================================")

st.write("Shinta Alya Imani Putri")
st.write("200411100005")

tab1, tab2, tab3, tab4 = st.tabs(["Import Data", "Preprocessing", "Modelling", "Evalutions"])

with tab1:
    st.write("Import Data")
    df = pd.read_csv("https://raw.githubusercontent.com/Shintaalya/Datafile/main/mushrooms.csv")
    st.dataframe(df)

with tab2:
    df.head()
    
    df.isnull().sum()
    
    st.write("Mengubah Char Label Menjadi Numerik")
    df = df.apply(lambda col: pd.factorize (col, sort=True)[0])
    df
    
    st.write("Keterangan :")
    st.write("1 = tidak dapat dimakan")
    st.write("0 = dapat dimakan")
 
with tab3:
    st.write(""#Metode"")
    st.write("1. KNN")
    
    st.write("Pembagian x dan y")
    X = df.iloc[:,1:].values
    y = df.iloc[:,0].values 
    
    st.write("#mencari K terbaik (1-10) dulu")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=3)

    print("K\tAkurasi")
    list_k = []

    for i in range(1,11):
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(X_train, y_train)
        acc = classifier.score(X_test, y_test)
        list_k.append(acc)
        print(str(i)+"\t"+str(acc))

    print()
    print("Akurasi tertinggi \t: "+str(max(list_k)))
    tertinggi = list_k.index(max(list_k))+1
    print("Berarti K nya adalah \t: "+(str(tertinggi)))
