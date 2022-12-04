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
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # X_train = cv.fit_transform(X_train)
    # X_test = cv.fit_transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)


    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)
