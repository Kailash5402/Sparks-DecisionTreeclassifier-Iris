#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import sklearn as sk
data = pd.read_csv("Iris.csv")
x = data
from sklearn import preprocessing as pp
le = pp.LabelEncoder()
y = le.fit_transform(data['Species'])
data['Type']=y
from sklearn.model_selection import train_test_split
features = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
types = data["Type"]
train_features, test_features, train_types, test_types = train_test_split(features,types,train_size=0.8, random_state=1)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_features, train_types)
def prediction(x):
    prediction = clf.predict(x)
    if prediction == 0:
        print("Iris setosa")
    elif prediction == 1:
            print("Iris versicolor")
    elif prediction == 2:
            print("Iris virginica")
print("Hello welcome to the Iris flower classifier.\nKindly enter the necessary data")
sl=input("Enter Sepal length in cm:")
sw=input("Enter Sepal Width in cm:")
pl=input("Enter petal length in cm:")
pw=input("Enter petal Width in cm:")
inp_data = pd.DataFrame({"SepalLengthCm" : [sl],"SepalWidthCm" : [sw],"PetalLengthCm" : [pl],"PetalWidthCm":[pw]})
#inp_data.insert(0,"SepalLengthCm", sl, allow_duplicates = False)
#inp_data.insert(1,"SepalWidthCm", sw, allow_duplicates = False)
#inp_data.insert(2,"PetalLengthCm", pl, allow_duplicates = False)
#inp_data.insert(3,"PetalWidthCm", pw, allow_duplicates = False)
a=inp_data.get("SepalLengthCm")
pd.to_numeric(a, errors='coerce')
b=inp_data.get("SepalWidthCm")
pd.to_numeric(a, errors='coerce')
c=inp_data.get("PetalLengthCm")
pd.to_numeric(a, errors='coerce')
d=inp_data.get("PetalWidthCm")
pd.to_numeric(a, errors='coerce')
dat = pd.DataFrame({"SepalLengthCm":[a],"SepalWidthCm":[b],"PetalLengthCm":[c],"PetalWidthCm":[d]})
input(print("Predict? (Y/N)"))
if "Y"or"y":
    prediction(dat)
else:
    exit()


# In[ ]:




