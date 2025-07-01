#polynomial
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('/content/drive/MyDrive/Dataset/energydata_complete_v1-1.csv')

data["date"] = pd.to_datetime(data["date"])
data["Time_alone"] = pd.to_datetime(data["Time_alone"], format='%H:%M:%S')
data["SFM"] = (data["date"] - data["date"].dt.normalize()).dt.total_seconds()
data['Date_alone'] = pd.to_datetime(data['Date_alone'])
data['DOM'] = (data['Date_alone'] - pd.to_datetime("2016-01-10")).dt.days
display_data = False
if display_data == True:
  # Display the first few rows of the dataset
  with pd.option_context('display.max_rows', 200,
                        'display.max_columns', 1,
                        'display.precision', 3,
                        ):
    print(data["SFM"])
    print(data["date"])
    print(data["date"] - data["date"].dt.normalize())
  print(data.head())
  # Summary statistics
  print(data.describe())

  # Check for missing values
  print(data.isnull().sum())
  # Making Selection of features data


data_sf = data[["T2",'RH_out','lights',"Appliances"]]
#print(data_SF.head())

data_pca_prep = data.drop(columns = ['date', 'Time_alone','SFM','Date_alone'])
pca = PCA().set_output(transform="pandas").fit(data_pca_prep)
data_pca = pca.transform(data_pca_prep)
data_pca = data_pca[['pca0','pca1','pca2','pca3']]

# Prepare the data without anything done to it
x = data.drop(['date', 'Time_alone','Date_alone','SFM'], axis=1)
y = data[['SFM']]

# spliting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4,random_state=1)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1)


pca_train, pca_test, y_train, y_test = train_test_split(data_pca , y, test_size=0.4,random_state=1)

pca_test, pca_val, y_test, y_val = train_test_split(pca_test, y_test, test_size=0.5, random_state=1)


sf_train, sf_test, y_train, y_test = train_test_split(data_sf , y, test_size=0.4,random_state=1)

sf_test, sf_val, y_test, y_val = train_test_split(sf_test, y_test, test_size=0.5, random_state=1)

mp_x = x.sample(frac=0.5,random_state=1)
mp_y = y.sample(frac=0.5,random_state=1)
mp_x_train, mp_x_test, mp_y_train, mp_y_test = train_test_split(mp_x, mp_y, test_size=0.4,random_state=1)

mp_x_test,mp_x_val, mp_y_test, mp_y_val = train_test_split(mp_x_test, mp_y_test, test_size=0.5, random_state=1)

model=Pipeline([('scaler',StandardScaler()),('poly', PolynomialFeatures()),('linear', LinearRegression())])


parameters={
    'poly__degree':[2,3,4,5]}


grid_search = GridSearchCV(model, parameters, cv=5, scoring='neg_root_mean_squared_error')
#grid_search.fit(x_train, y_train)
grid_search.fit(pca_train, y_train)
#grid_search.fit(sf_train, y_train)

#predictions = grid_search.predict(x_test)
predictions = grid_search.predict(pca_test)
#predictions = grid_search.predict(sf_test)

print(grid_search.best_params_)
print(grid_search.best_estimator_)

model=Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression())])
#model.fit(x_train, y_train)
model.fit(pca_train, y_train)
#model.fit(sf_train, y_train)
#cv=3
cv=5
#cv=10
#y_train_pred=model.predict(x_train)
#y_train_pred=model.predict(pca_train)
y_train_pred=model.predict(sf_train)
#scores=cross_validate(model, x_train, y_train, cv=cv, scoring=['neg_root_mean_squared_error','neg_mean_absolute_error','r2'])
scores=cross_validate(model, pca_train, y_train, cv=cv, scoring=['neg_root_mean_squared_error','neg_mean_absolute_error','r2'])
#scores=cross_validate(model, sf_train, y_train, cv=cv, scoring=['neg_root_mean_squared_error','neg_mean_absolute_error','r2'])
pos_mrse=-scores['test_neg_root_mean_squared_error']
mean_mrse=pos_mrse.mean()
pos_mae=-scores['test_neg_mean_absolute_error']
mean_mae=pos_mae.mean()
pos_r2=scores['test_r2']
mean_r2=pos_r2.mean()
print("regression mean_mrse:", mean_mrse)
print("regression mean_mae:", mean_mae)
print("regression mean_r2:", mean_r2)
