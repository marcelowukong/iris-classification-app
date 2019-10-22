# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:13:08 2019
@author: Wukong
"""

#Classificador Baseado em Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from pathlib import Path
import pandas as pd

#Definindo Paths
PATH_MODELS = Path("../../models")
PATH_DATA_RAW = Path("../../data-source/raw")

#Carragando Iris Dataset
#iris = load_iris()
iris = pd.read_csv(PATH_DATA_RAW/"raw_iris_data.csv")
X = iris.iloc[:,:-1]
y = iris.iloc[:,-1]

#Separando Conjunto de Dados
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    test_size=0.5)

#Crian Classificador
clf = RandomForestClassifier(n_estimators=10)

#Treinando Classificador
clf.fit(X_train, y_train)

#Predizendo com Classificador
predicted = clf.predict(X_test)

#Acuracia do Classificador
print(accuracy_score(predicted, y_test))

#Salvando Modelo
with open(PATH_MODELS/'rf.pkl', 'wb') as model_pkl:
    pickle.dump(clf, model_pkl)