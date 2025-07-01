# PLOTS FOR NO SF DATA
corr_matrix = data.corr()
plt.figure(figsize=(25, 25))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Feature importance analysis using a simple regression model (e.g., Random Forest)
from sklearn.ensemble import RandomForestRegressor

# Initialize and fit the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x, y)

# Get feature importances
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()