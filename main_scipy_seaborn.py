import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Seaborn for advanced visualization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('housing_data.csv')
print(df.head())

# --- Seaborn Visualization: Pairplot ---
# This shows pairwise relationships in the dataset, colored by price
sns.pairplot(df, diag_kind='kde')
plt.suptitle('Pairwise Feature Relationships', y=1.02)
plt.show()

# --- Seaborn Visualization: Correlation Heatmap ---
# This shows how features are correlated with each other and with price
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

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
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# We use fit_transform on training data to learn scaling parameters, and transform on test data to apply the same scaling.

# Initialize and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: ${mse:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")

# --- Seaborn Visualization: Actual vs Predicted ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

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

# Scale new data using the same scaler
new_houses_scaled = scaler.transform(new_houses)
new_predictions = model.predict(new_houses_scaled)

print("New house predictions:")
for i, pred in enumerate(new_predictions):
    print(f"House {i+1}: ${pred:,.2f}")