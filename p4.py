# final model testing
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
from sklearn.metrics import r2_score

#data=pd.read_csv('/content/drive/MyDrive/Dataset/energydata_complete_v1-1.csv')

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

#test
fd_x_train, fd_x_test, fd_y_train, fd_y_test = train_test_split(x, y, test_size=0.01,random_state=1)
'''
reg = Pipeline([("scaler", StandardScaler()),("mlp", MLPRegressor(hidden_layer_sizes=(20, 15,5), random_state=1, max_iter= 5000,
solver='lbfgs'))])

x_train_merged=pd.concat([x_train, x_val])
y_train_merged=pd.concat([y_train, y_val])

reg.fit(x_train_merged, np.ravel(y_train_merged))
y_pred_train_merged = reg.predict(x_train_merged)
y_pred_test = reg.predict(x_test)

neg_rmse_train=-np.sqrt(np.mean((y_pred_train_merged - np.ravel(y_train_merged))**2))
neg_rmse_test=-np.sqrt(np.mean((y_pred_test - np.ravel(y_test))**2))
pos_rmse_train=-neg_rmse_train
pos_rmse_test=-neg_rmse_test
print("pos_rmse_train:", pos_rmse_train)
print("pos_rmse_test:", pos_rmse_test)

neg_mae_train=-np.mean(np.abs(y_pred_train_merged - np.ravel(y_train_merged)))
neg_mae_test=-np.mean(np.abs(y_pred_test - np.ravel(y_test)))
pos_mae_train=-neg_mae_train
pos_mae_test=-neg_mae_test
print("pos_mae_train:", pos_mae_train)
print("pos_mae_test:", pos_mae_test)

r2_train = r2_score(np.ravel(y_train_merged), y_pred_train_merged)  # Corrected line
r2_test = r2_score(np.ravel(y_test), y_pred_test)  # Corrected line
print("r2_train:", r2_train)
print("r2_test:", r2_test)
'''
reg = Pipeline([("scaler", StandardScaler()),("mlp", MLPRegressor(hidden_layer_sizes=(19, 13,11), random_state=1, max_iter= 1000,
solver='lbfgs'))])

x_train_merged=pd.concat([x_train, x_val])
y_train_merged=pd.concat([y_train, y_val])

reg.fit(fd_x_train, np.ravel(fd_y_train))
y_pred_train_merged = reg.predict(fd_x_train)
y_pred_test = reg.predict(fd_x_test)

neg_rmse_train=-np.sqrt(np.mean((y_pred_train_merged - np.ravel(fd_y_train))**2))
neg_rmse_test=-np.sqrt(np.mean((y_pred_test - np.ravel(fd_y_test))**2))
pos_rmse_train=-neg_rmse_train
pos_rmse_test=-neg_rmse_test
print("pos_rmse_train:", pos_rmse_train)
print("pos_rmse_test:", pos_rmse_test)

neg_mae_train=-np.mean(np.abs(y_pred_train_merged - np.ravel(fd_y_train)))
neg_mae_test=-np.mean(np.abs(y_pred_test - np.ravel(fd_y_test)))
pos_mae_train=-neg_mae_train
pos_mae_test=-neg_mae_test
print("pos_mae_train:", pos_mae_train)
print("pos_mae_test:", pos_mae_test)

r2_train = r2_score(np.ravel(fd_y_train), y_pred_train_merged)  # Corrected line
r2_test = r2_score(np.ravel(fd_y_test), y_pred_test)  # Corrected line
print("r2_train:", r2_train)
print("r2_test:", r2_test)