# split/pca maybe the jump is to hight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.preprocessing import (MinMaxScaler,StandardScaler)
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import PredictionErrorDisplay
from itertools import product


sns.set_theme()

data = pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/energydata_complete_v1.csv')
data.head()

#data = data.drop('Date_alone',axis=1)
# Pre prep of data
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


#Meta Parameters data for "speed"
mp_x = x.sample(frac=0.5,random_state=1)
mp_y = y.sample(frac=0.5,random_state=1)
mp_x_train, mp_x_test, mp_y_train, mp_y_test = train_test_split(mp_x, mp_y, test_size=0.4,random_state=1)

mp_x_test,mp_x_val, mp_y_test, mp_y_val = train_test_split(mp_x_test, mp_y_test, test_size=0.5, random_state=1)

reg = Pipeline([("scaler", StandardScaler()),("mlp", MLPRegressor(hidden_layer_sizes=(20, 15,5), random_state=1, max_iter= 5000,
solver='lbfgs'))])

first_layer_neurons = np.arange(1, 20, 5)
second_layer_neurons = np.arange(1, 20, 5)
third_layer_neurons = np.arange(1, 20, 5)
hidden_layer_sizes = list(product(first_layer_neurons, second_layer_neurons,third_layer_neurons))
# dictionary of parameters: takes all combinations
parameters_1 = {
    'mlp__hidden_layer_sizes':  hidden_layer_sizes,
    'mlp__solver': ['lbfgs'],
    #'mlp__max_iter': [100, 500, 1000]
}

#Best from [i,i] was [5,5] -9138
#best from [i,i,i] was []

# OR a sequence of dictionaries of parameters; usefull to avoid unnecessary parameters combination

for parameters in [parameters_1]:
  grid_search = GridSearchCV(reg, parameters, cv=2, scoring='neg_mean_absolute_error')
  grid_search.fit(mp_x_train, mp_y_train)
  print(grid_search.best_params_) # to get the best parameters
  print(grid_search.best_estimator_) # to get the best estimator
  print(grid_search.cv_results_) # to get all results
  cv_res = pd.DataFrame(grid_search.cv_results_) # or use DataFrame
  cv_res.to_excel("Wyniki.xlsx")
  display(cv_res) # to display results as a table