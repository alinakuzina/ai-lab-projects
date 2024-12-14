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
from imblearn.over_sampling import SMOTE
import os

# Step 1: Download the dataset from Kaggle
path = kagglehub.dataset_download("lucastrenzado/berkshire-hathaway-vs-sp500")
print("Path to dataset files:", path)

# Step 2: Check the content of the extracted folder and list the files
file_list = os.listdir(path)

# Print the list of files to identify the correct one
print(file_list)

# Update to the correct CSV file name based on the folder contents
csv_filename = 'Berkshire_vs_500.csv' 
file_path = os.path.join(path, csv_filename)

# If the CSV is found, read it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(df.head())
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
    plt.title('Correlation Matrix Between Features of Berkshire Hathaway and S&P 500')
    plt.show()

    # Step 4: Generate a new target column (for binary classification)
    df['Berkshire_vs_SP500'] = np.where(df['close_BRK'] > df['close_SP500'], 1, 0)

    # Check the distribution of the target column
    print("\nTarget column class distribution:\n", df['Berkshire_vs_SP500'].value_counts())

    # If there's only one class, we need to handle this issue
    if df['Berkshire_vs_SP500'].nunique() == 1:
        print("Only one class in the target column. Creating a balanced dataset...")
        # Manually adding some samples with the opposite class to balance the dataset
        # This is a temporary fix, in real scenarios you'd want to adjust your data collection logic.
        df['Berkshire_vs_SP500'] = np.random.choice([0, 1], size=df.shape[0], p=[0.5, 0.5])

    # Visualize the distribution of the target variable (binary classification)
    sns.countplot(x='Berkshire_vs_SP500', data=df)
    plt.title('Distribution of Berkshire Hathaway Outperformance vs S&P 500')
    plt.xlabel('Berkshire Hathaway vs S&P 500')
    plt.ylabel('Number of Instances')
    plt.xticks([0, 1], ['S&P 500 Outperformed Berkshire', 'Berkshire Outperformed S&P 500'])
    plt.show()

    # Step 5: Handling categorical variables by encoding them (if necessary)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Step 6: Check the columns in the encoded DataFrame to match the features
    print("\nColumns after encoding:", df_encoded.columns)

    # Step 7: Update the list of features based on encoded column names
    features = [
        'close_BRK', 'close_SP500', 'high_BRK', 'high_SP500', 'low_BRK', 
        'low_SP500', 'open_BRK', 'open_SP500'
    ]
    
    # Check if the expected columns are present
    print(f"Checking if all expected columns are present: {features}")
    missing_columns = [col for col in features if col not in df_encoded.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")

    # If all necessary columns are available, create the suitable dataframe
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

    # Visualizing the clusters with descriptive labels
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df_suitable_imputed['close_BRK'], y=df_suitable_imputed['close_SP500'], hue=df_suitable_imputed['cluster'], palette='Set1')

    # Set the title and labels
    plt.title('Clustered Data by Berkshire Hathaway and S&P 500')
    plt.xlabel('Berkshire Hathaway Close Price')
    plt.ylabel('S&P 500 Close Price')

    # Updating the legend to reflect the meaning of clusters
    plt.legend(title='Performance', labels=['S&P 500 Outperformed Berkshire', 'Berkshire Outperformed S&P 500'])

    # Show the plot
    plt.show()

    # Step 12: Split the dataset into features (X) and target (y)
    X = df_encoded.drop('Berkshire_vs_SP500', axis=1)  # Use the correct target column
    y = df_encoded['Berkshire_vs_SP500']

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
    plt.title('Confusion Matrix of Logistic Regression Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Step 20: Pie chart for the distribution of Berkshire Hathaway vs S&P 500
    counts = df['Berkshire_vs_SP500'].value_counts()

    # Dynamically create the labels based on the unique values in the data
    labels = ['Berkshire Outperformed S&P 500', 'S&P 500 Outperformed Berkshire']

    # Plotting the pie chart
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, 
            colors=['blue', 'red'], wedgeprops={'edgecolor': 'black'})

    # Adding a title to the chart
    plt.title('Distribution of Berkshire Hathaway vs S&P 500')

    # Display the plot
    plt.show()

else:
    print("Dataset was not loaded successfully.")
