# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 22:12:06 2021

@author: hüseyin aksak
"""
#kullanılan kutuphaneler import edildi

import pandas as pd
from sklearn import preprocessing 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
import numpy as np
from sklearn import model_selection
from sklearn.metrics import mean_squared_error as MSE

#veri dataframe olarak okundu

df = pd.read_csv('voks.csv', encoding="utf-8") #df olarak okuduk

# veri setini incelemek için işlemler..
"""
print(df.info())
print(df.describe())
print(df.Seri.value_counts())
print(df.Model.value_counts())
print(df.Yıl.value_counts())
print(df.Km.value_counts())
print(df.Renk.value_counts())
print(df.Vites.value_counts())
print(df.Yakıt.value_counts())
print(df.Sehir.value_counts())
print(df.Tarih.value_counts())
"""
#işimize yaramayan sütunları sildik(ilanlar 1-2 aylık süre içinde olduğu için tarih değikeni de silindi) 

df.drop("Unnamed: 0", axis=1, inplace=True)
df.drop("_id",axis = 1,inplace=True)
df.drop("Id",axis = 1,inplace=True)
df.drop("Marka",axis = 1,inplace=True)
df.drop("Tarih",axis = 1,inplace=True)

#kategorik değişkenler etiketledik
le = preprocessing.LabelEncoder()
df.dropna(subset = ["Sehir"], inplace=True)
df["Sehir"] = le.fit_transform(df.Sehir)
df["Model"] = le.fit_transform(df.Model)
df["Seri"] = le.fit_transform(df.Seri)
df["Renk"] = le.fit_transform(df.Renk)
df["Vites"] = le.fit_transform(df.Vites)
df["Yakıt"] = le.fit_transform(df.Yakıt)
df["Fiyat"] = df.Fiyat.str[:-3]
#print(le.classes_)
#print(df.Sehir.unique())
#print(le.inverse_transform([37]))

X = df.drop(["Fiyat"], axis = 1)
y = df["Fiyat"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

xgb = XGBRegressor()

#xgboost icin en iyi parametleri buluyor,çalışması uzun sürüyor

"""
params = {"colsample_bytree":[0.4,0.5,0.6],
         "learning_rate":[0.01,0.02,0.09],
         "max_depth":[2,3,4,5,6],
         "n_estimators":[100,200,500,2000]}

grid = GridSearchCV(xgb, params, cv = 10, n_jobs = -1, verbose = 2)
grid.fit(X_train, y_train)
print(grid.best_params_)
"""
#parametreleri girip model eğitildi
xgb1 = XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.02, max_depth = 4, n_estimators = 2000)
model_xgb = xgb1.fit(X_train, y_train)
pred=model_xgb.predict(X_test)

#seçilen 5 kayıtın gerçek değerleri ve tahmin değerleri
print(model_xgb.predict(X_test)[57:62])
print(y_test[57:62])

#modelin skorunu hesaplandı
print(model_xgb.score(X_test, y_test))
print(model_xgb.score(X_train, y_train))

#attributeların önem derecesi
importance = pd.DataFrame({"Importance": model_xgb.feature_importances_},index=X_train.columns)
print(importance)


#RMSE ve MSE Skorları
rmse = np.sqrt(MSE(y_test, pred))
print(rmse)
print(MSE(y_test, pred))

















