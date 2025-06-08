import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Select features and target
features = ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'FullBath']
X = train_df[features]
y = train_df['SalePrice']

# 3. Handle missing values (if any)
X = X.fillna(X.mean())
y = y.fillna(y.mean())
X_test = test_df[features].fillna(X.mean())

# 4. Split train data for evaluation (optional but recommended)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Print coefficients and intercept
print("Intercept (b0):", model.intercept_)
for feature, coef in zip(features, model.coef_):
    print(f"Coefficient for {feature}:", coef)

# 7. Evaluate the model on validation set
y_pred = model.predict(X_val)
print("\nModel Performance on Validation Set:")
print("R^2 Score:", r2_score(y_val, y_pred))
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print('Root mean squared Error:', rmse)

# 8. Predict on test data
test_predictions = model.predict(X_test)

# 9. Save predictions (optional)
output = test_df.copy()
output['PredictedPrice'] = test_predictions
output.to_csv('house_price_predictions.csv', index=False)
print("\nPredictions saved to 'house_price_predictions.csv'")