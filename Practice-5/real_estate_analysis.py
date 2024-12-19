import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Step 1: Download the dataset
path = kagglehub.dataset_download("kirichenko17roman/kyiv-real-estate")

# Print the path to dataset files
print("Path to dataset files:", path)

# Step 2: List the contents of the directory where the dataset was downloaded
print("Listing contents of dataset directory:", path)
print(os.listdir(path))

# Step 3: Load the correct dataset (class_flat.csv)
dataset_file_path = path + '/class_flat.csv'

if os.path.exists(dataset_file_path):
    df = pd.read_csv(dataset_file_path)
    print("Dataset loaded successfully!")
    print(df.head())  
    print("\nColumns in the dataset:", df.columns) 
else:
    print(f"Error: {dataset_file_path} not found.")
    exit()

# Step 4: Data Preprocessing

# a) Handling missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# You can fill missing values or drop them depending on your approach
df_cleaned = df.dropna() 

# b) Update categorical columns (based on the available columns)
# We will use the actual column names from the dataset instead of 'property_type' and 'city'
# Replace 'property_type' and 'city' with valid categorical column names from the printed list

df_encoded = pd.get_dummies(df_cleaned, columns=['complex', 'city'], drop_first=True) 

# c) Normalize numerical features
scaler = StandardScaler()
df_encoded[['price', 'rooms', 'area_total']] = scaler.fit_transform(df_encoded[['price', 'rooms', 'area_total']])

# Step 5: Exploratory Data Analysis (EDA)

# Ensure that we only select numeric columns for correlation calculation
numeric_columns = df_encoded.select_dtypes(include=[np.number]).columns

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.show()

# Scatter plot for Area vs Price
sns.scatterplot(x=df_encoded['area_total'], y=df_encoded['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()

# Step 6: Build Simple Linear Regression Model (one feature: area)

X = df_encoded[['area_total']]
y = df_encoded['price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Visualize Actual vs Predicted Prices for Simple Linear Regression
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Simple Linear Regression: Actual vs Predicted Prices')
plt.show()

# Step 7: Build Multiple Linear Regression Model (multiple features)

X_multi = df_encoded[['area_total', 'rooms']]
y_multi = df_encoded['price']

# Split data into training and test sets
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Create and train the multiple linear regression model
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

# Make predictions
y_pred_multi = model_multi.predict(X_test_multi)

# Evaluate the model
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
mae_multi = mean_absolute_error(y_test_multi, y_pred_multi)

print(f'Mean Squared Error (MSE) for Multiple Regression: {mse_multi}')
print(f'Mean Absolute Error (MAE) for Multiple Regression: {mae_multi}')

# Visualize Actual vs Predicted Prices for Multiple Linear Regression
plt.scatter(y_test_multi, y_pred_multi)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Multiple Linear Regression: Actual vs Predicted Prices')
plt.show()

# Step 8: Save the Results
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_results.to_csv('simple_linear_regression_results.csv', index=False)

df_results_multi = pd.DataFrame({'Actual': y_test_multi, 'Predicted': y_pred_multi})
df_results_multi.to_csv('multiple_linear_regression_results.csv', index=False)

print("Results saved to CSV files.")
