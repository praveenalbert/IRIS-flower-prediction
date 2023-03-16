import pandas as pd
import numpy as np
import pickle

df=pd.read_csv(r"D:\users\Praveen kumar\pycharmprojects\exflask\__pycache__\IRIS.csv")


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

y=le.fit_traynsform(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.svm import SVC
sv=SVC(kernel='linear').fit(X_train,y_train)


pickle.dump(sv,open('iri.pkl','wb'))