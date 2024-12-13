import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import os

# Step 1: Download the dataset using kagglehub
path = kagglehub.dataset_download("redwankarimsony/heart-disease-data")
print("Path to dataset files:", path)

# Step 2: List the files in the extracted dataset folder to find the correct CSV file
extracted_path = os.path.join(path, 'extracted_data')  # Check if the file is in 'extracted_data' folder
if not os.path.exists(extracted_path):  # If the folder doesn't exist, check the root folder
    extracted_path = path

# List all files in the dataset directory
print("Listing contents of the dataset folder:", extracted_path)
file_list = os.listdir(extracted_path)

# Print all files to identify the correct CSV file
print(file_list)

# Check for the correct file name
csv_filename = 'heart_disease_uci.csv'  # The correct file name based on the listing
if csv_filename not in file_list:
    print(f"'{csv_filename}' not found. Available files: {file_list}")
    # You can manually choose another file if necessary or update the filename here.

# Construct the correct file path
file_path = os.path.join(extracted_path, csv_filename)

# Ensure the file exists before attempting to read it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(df.head())  # Display the first few rows of the dataset
else:
    print(f"File {csv_filename} not found at {file_path}")

# Step 3: Data Preprocessing

# Check for missing values in the dataset
if 'df' in locals():
    # Print missing data information
    print("\nMissing data check:\n", df.isnull().sum())
    
    # Statistical summary of the dataset
    print("\nDataset Summary:\n", df.describe())

    # Drop non-numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])

    # Visualize the correlation matrix to identify relationships between features
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    # Step 4: Handle the target column properly, check if it's renamed after encoding
    target_column = 'num'  # After encoding, the target column is 'num'
    if target_column not in df.columns:
        print(f"Expected column '{target_column}' not found. Available columns: {df.columns}")

    # Visualize the distribution of the target variable (presence of heart disease)
    sns.countplot(x=target_column, data=df)
    plt.title('Distribution of Heart Disease Presence')
    plt.show()

    # Step 5: Handling categorical variables by encoding them
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Step 6: Check the columns in the encoded DataFrame to match the features
    print("\nColumns after encoding:", df_encoded.columns)

    # Step 7: Update the list of features based on encoded column names
    features = [
        'age', 'sex_Male', 'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina', 
        'trestbps', 'chol', 'fbs_True', 'restecg_normal', 'restecg_st-t abnormality', 
        'thalch', 'exang_True', 'slope_flat', 'slope_upsloping', 'thal_normal', 'thal_reversable defect', 'ca'
    ]
    
    # Check if 'thalch' is present in the columns after encoding
    print(f"Checking if all expected columns are present: {features}")
    missing_columns = [col for col in features if col not in df_encoded.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")

    # If 'thalch' or other necessary columns are missing, check their actual names in the dataframe
    df_suitable = df_encoded[features]

    # Step 8: Check for NaN values in df_suitable before clustering
    print("\nChecking for missing values in the suitable features dataframe:")
    print(df_suitable.isnull().sum())

    # Step 9: Impute missing values using SimpleImputer (for numeric columns)
    imputer = SimpleImputer(strategy='mean')  # Use mean for numeric columns
    df_suitable_imputed = pd.DataFrame(imputer.fit_transform(df_suitable), columns=df_suitable.columns)

    # Step 10: Check if there are still any NaN values
    print("\nChecking for missing values after imputation:")
    print(df_suitable_imputed.isnull().sum())

    # Step 11: Clustering using KMeans (regression-based clustering method)
    kmeans = KMeans(n_clusters=2, random_state=42)
    df_suitable_imputed['cluster'] = kmeans.fit_predict(df_suitable_imputed)

    # Assign meaningful labels to the clusters based on characteristics
    df_suitable_imputed['cluster_label'] = df_suitable_imputed['cluster'].apply(
        lambda x: 'Low Risk' if x == 0 else 'High Risk'
    )

    # Visualizing the clusters (checking if 'age' and 'chol' exist in the final dataset)
    if 'age' in df_suitable_imputed.columns and 'chol' in df_suitable_imputed.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=df_suitable_imputed['age'], y=df_suitable_imputed['chol'], hue=df_suitable_imputed['cluster_label'], palette='Set1')
        plt.title('Clustered Data by Age and Cholesterol')
        plt.xlabel('Age')
        plt.ylabel('Cholesterol')
        plt.legend(title='Risk Level', loc='upper right')
        plt.show()
    else:
        print("Columns 'age' or 'chol' are missing from the dataset.")

    # Step 12: Split the dataset into features (X) and target (y)
    X = df_encoded.drop(target_column, axis=1)  # Use the correct target column
    y = df_encoded[target_column]

    # Step 13: Impute missing values in X (features)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Step 14: Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Step 15: Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 16: Build the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Step 17: Make predictions on the test data
    y_pred = model.predict(X_test_scaled)

    # Step 18: Evaluate the model using classification metrics
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Step 19: Visualize the confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Step 20: Pie chart for the distribution of heart disease presence
    heart_disease_counts = df[target_column].value_counts()

    # Dynamically create the labels based on the unique values in the data
    labels = ['No Heart Disease (0)', 'Mild Heart Disease (1)', 'Moderate Heart Disease (2)', 
              'Severe Heart Disease (3)', 'Very Severe Heart Disease (4)']

    # Plotting the pie chart
    plt.figure(figsize=(6,6))
    plt.pie(heart_disease_counts, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['blue', 'red', 'green', 'orange', 'purple'], wedgeprops={'edgecolor': 'black'})

    # Adding a title to the chart
    plt.title('Distribution of Heart Disease Presence')

    # Adding a legend to describe what the colors represent
    plt.legend(title="Heart Disease Presence", labels=labels, loc='upper left')

    # Display the plot
    plt.show()

else:
    print("Dataset was not loaded successfully.")
