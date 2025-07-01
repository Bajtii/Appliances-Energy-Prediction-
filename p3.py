solver='lbfgs'
hidden_layer_sizes=(19,13,11)
max_iter = 2000
reg = Pipeline([("scaler", StandardScaler()),("mlp", MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter,
solver=solver))])

#VERY DANGEROUS BE CAREFULL!!
# in case of regression , R^2 mean_absolute_error
scores = cross_validate(reg, fd_x_train, fd_y_train, cv=10,scoring=['r2','neg_mean_absolute_error'])
print('test_r2')
print(scores['test_r2'])
print('test_neg_mean_absolute_error')
print(-1*scores['test_neg_mean_absolute_error'])
print('fit_time')
print(scores['fit_time'])
print('score_time')
print(scores['score_time'])

