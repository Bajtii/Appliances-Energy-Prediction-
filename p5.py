
for max_iter in [100, 500, 1000,2000,5000,10000]:
  reg = Pipeline([("scaler", StandardScaler()),("mlp", MLPRegressor(hidden_layer_sizes=(19, 13,11), random_state=1, max_iter=max_iter,
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