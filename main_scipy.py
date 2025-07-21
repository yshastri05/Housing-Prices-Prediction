import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')



# Load the dataset
df = pd.read_csv('housing_data.csv')

print(df.head())





# Prepare data for modeling
X = df.drop('price', axis=1)
y = df['price']


print("x")
print(X.head())
print("y")
print(y.head())


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Scale features only for Linear Regression
# Random Forest does not require scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test) # why not use fit and transform?



# Initialize models
model = LinearRegression()
results = {}


model.fit(X_train_scaled, y_train) # this is why we scale the features and train the model
y_pred = model.predict(X_test_scaled) # this is why we scale the features and make predictions

    
# Calculate metrics 
mse = mean_squared_error(y_test, y_pred) # Mean Squared Error is used to measure the average of the squares of the errors
rmse = np.sqrt(mse) # Root Mean Squared Error is the square root of the mean of the squared errors
mae = mean_absolute_error(y_test, y_pred) # Mean Absolute Error is the average of the absolute differences between predictions and actual values

results["LinearRegression"] = {
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'predictions': y_pred
}

print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")


# Make predictions on new data
print("\n--- Making Predictions on New Houses ---")
new_houses = pd.DataFrame({
    'bedrooms': [3, 4, 2],
    'bathrooms': [2, 3, 1],
    'sqft_living': [2000, 3500, 1200],
    'sqft_lot': [8000, 12000, 5000],
    'floors': [2, 2, 1],
    'waterfront': [0, 1, 0],
    'condition': [4, 5, 3],
    'grade': [8, 10, 6],
    'yr_built': [2000, 2010, 1980]
})

# Use linear regression for predictions 
new_predictions = model.predict(new_houses)

print("New house predictions:")
for i, pred in enumerate(new_predictions):
    print(f"House {i+1}: ${pred:,.2f}")

