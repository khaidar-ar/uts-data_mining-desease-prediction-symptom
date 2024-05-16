
# Import dependencies in python
* pandas untuk library dataframe
* numpy library untuk pengolahan array
* seaborn library untuk pengolahan grafik statistik
* matplotlib library untuk grafik plotting
```
import csv
import pandas as pd
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
 ```   
# Memuat dataset menggunakan pandas
df merupakan variabel untuk memanggil fungsi pandas dengan method `read_excel` untuk membaca file dari local berformal xlsx.
 ```
 df = pd.read_excel('./dataset/raw_data.xlsx')
```
# Menampilkan record awal
 Memanggil method `head()` untuk menampilkan beberapa record data pertama dari dataset.
 ```
 df.head()
 ```
# Mengisi data kosong
Mengisi data kosong dengan memanggil method `fillna(method='ffill')`
```
data = df.fillna(method='ffill')
```
# Splitting format penamaan record data
```
def process_data(data):
    data_list = []
    data_name = data.replace('^','_').split('_')
    n = 1
    for names in data_name:
        if (n % 2 == 0):
            data_list.append(names)
        n += 1
    return data_list
```
# Data cleaning
```
disease_list = []
disease_symptom_dict = defaultdict(list)
disease_symptom_count = {}
count = 0

for idx, row in data.iterrows():
    
    # Get the Disease Names
    if (row['Disease'] !="\xc2\xa0") and (row['Disease'] != ""):
        disease = row['Disease']
        disease_list = process_data(data=disease)
        count = row['Count of Disease Occurrence']

    # Get the Symptoms Corresponding to Diseases
    if (row['Symptom'] !="\xc2\xa0") and (row['Symptom'] != ""):
        symptom = row['Symptom']
        symptom_list = process_data(data=symptom)
        for d in disease_list:
            for s in symptom_list:
                disease_symptom_dict[d].append(s)
            disease_symptom_count[d] = count
```
# Invoke Variabel
Menampilkan data penyakit yang diinisiasi pada variabel `disease_symptom_dict`dengan tipe data dictionary.
```
disease_symptom_dict
```
# Print tipe data
Menampilkan tipe data setiap kolom attribute pada data frame dengan memanggil fungsi `dtypes.`
```
df.dtypes
```
# Label encoding
Proses pelabelan data pada attribut/kolom target dengan nama `symptom`dengan memanggil package `LabelEncoder` milik library scikitlearn dengan fungsinya `fit_transform`untuk merubah data kategorik menjadi numerik.
```
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['symptom'])
print(integer_encoded)
```
# One hot encoding
Merubah data numerik menjadi kategorik dengan memanggil package `OneHotEncoder` milik library *scikitlearn* dengan fungsinya `fit_transform`.
 ```
 onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
```
# Cek redundansi data
Melakukan pengecekan perulangan data yang sama pada kolom dengan memanggil fungsi `unique()` milik library *numpy* pada kolom *symptom*
```
cols = np.asarray(df['symptom'].unique())
cols
```
# Transpose baris menjadi nama kolom data frame
Merubah setiap record data pada attribute `symptom` menjadi nama kolom pada data frame.
```
df_ohe = pd.DataFrame(columns = cols)
df_ohe.head()
```
# Mapping kolom attribute dengan hasil pelabelan
```
for i in range(len(onehot_encoded)):
    df_ohe.loc[i] = onehot_encoded[i]
```
# Menggabungkan data frame awal dengan data frame hasil encoding
``` 
df_concat = pd.concat([df_disease,df_ohe], axis=1)
df_concat.head()
```
# Menghapus baris data yang sama
Menghapus duplikasi data berdasarkan kolom pertama `desease.`
```
df_concat.drop_duplicates(keep='first',inplace=True)
```
# Simpan dataset
Menyimpan dataframe pada lokal storage dengan format `csv`.
```
df_concat.to_csv("./dataset/training_dataset.csv", index=False)
```
# Import dependencies model
* `train_test_split` library untuk splitting dataset pada proses training dan testing
* `MultinomialNB` library untuk model naive bayes pada data yang bersifat diskrit
* `tree` library untuk model *decision tree*
* `DecisionTreeClassifier` library untuk model klasifikasi *decision tree*
* `export_graphviz` library untuk menampilkan graph dari *decision tree*
```
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
```
# Splitting dataset
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
```
# Cek jumlah data training
```
len(X_train), len(y_train)
```
# Cek jumlah data testing
```
len(X_test), len(y_test)
```
# Training model
Melatih model dengan memanggil package`DecisionTreeClassifier` dengan fungsi `fit()`
```
dt = DecisionTreeClassifier()
clf_dt=dt.fit(X, y)
```
# Akurasi model
Pengecekan akurasi model dengan memanggil fungsi `score`
```
clf_dt.score(X, y)
```
# Menginstall package graphviz
```
!pip  install  graphviz
```
# Menyimpan grafik decision tree
Menampilkan grafik model *decision tree* yang dihasilkan dengan memanggil fungsi `export_graphviz` dan disimpan pada file dengan nama `tree.dot` 
```
export_graphviz(dt, 
                out_file='./tree.dot', 
                feature_names=cols)
```
# Proses penampilan grafik decision tree
Menampilkan grafik model *decision tree* dengan package`Source` dan menyimpannya dengan format *png* secara *stream* dengan memanggil fungsi `pipe`
```
from graphviz import Source
from sklearn import tree

graph = Source(export_graphviz(dt, 
                out_file=None, 
                feature_names=cols))

png_bytes = graph.pipe(format='png')

with open('tree.png','wb') as f:
    f.write(png_bytes)
```
# Menampilkan grafik decision tree pada console
```
from IPython.display import Image
Image(png_bytes)
```
# Prediksi
Memprediksi dengan model *decision tree* yang dikembangkan dengan memanggil fungsi `predict`
```
disease_pred = clf_dt.predict(X)
```
# Inisiasi nilai aktual
```
disease_real = y.values
```
# Perbandingan nilai prediksi dan aktual
```
for i in range(0, len(disease_real)):
    if disease_pred[i]!=disease_real[i]:
        print ('Pred: {0}\nActual: {1}\n'.format(disease_pred[i], disease_real[i]))
```
