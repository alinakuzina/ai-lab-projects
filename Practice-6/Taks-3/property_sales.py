import kagglehub
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import os

# Step 1: Download the dataset from Kaggle
path = kagglehub.dataset_download("agungpambudi/property-sales-data-real-estate-trends")
print("Path to dataset files:", path)

# Step 2: Check the content of the extracted folder and list the files
file_list = os.listdir(path)
print(file_list)

# Update to the correct CSV file name based on the folder contents
csv_filename = '2002-2018-property-sales-data.csv'
file_path = os.path.join(path, csv_filename)

# If the CSV is found, read it
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(df.head())
else:
    print(f"File {csv_filename} not found at {file_path}")

# Step 3: Data Preprocessing
df_residential = df[df['PropType'] == 'Residential'].copy()  # Create a copy to avoid the SettingWithCopyWarning

# Check for missing values in the filtered dataset
if 'df_residential' in locals():
    # Print missing data information
    print("\nMissing data check for Residential properties:\n", df_residential.isnull().sum())
    
    # Statistical summary of the dataset
    print("\nDataset Summary for Residential properties:\n", df_residential.describe())

    # Add 'Year' column based on the 'Sale_date' column (corrected from 'date')
    df_residential['Year'] = pd.to_datetime(df_residential['Sale_date'], errors='coerce').dt.year

    # Step 4: Compare sales data across different years for Residential properties
    df_yearly = df_residential.groupby('Year').agg(
        avg_price=('Sale_price', 'mean'),
        total_sales=('Sale_price', 'count'),
        avg_area=('Fin_sqft', 'mean')  # Changed from 'area' to 'Fin_sqft'
    ).reset_index()

    # Visualize the average price by year
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='avg_price', data=df_yearly, marker='o', color='blue')
    plt.title('Average Residential Property Price by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.xticks(rotation=45)
    plt.show()

    # Visualize the total sales by year
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='total_sales', data=df_yearly, marker='o', color='green')
    plt.title('Total Residential Property Sales by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.show()

    # Visualize the average property area by year
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y='avg_area', data=df_yearly, marker='o', color='red')
    plt.title('Average Residential Property Area by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Area (sqft)')
    plt.xticks(rotation=45)
    plt.show()

    # Step 5: Generate a new target column (for binary classification)
    df_residential['price_class'] = np.where(df_residential['Sale_price'] > df_residential['Sale_price'].median(), 1, 0)

    # Step 6: Handle categorical variables by encoding them
    df_encoded = pd.get_dummies(df_residential, drop_first=True)

    # Step 7: Select the features for clustering and classification
    features = [
        'Fin_sqft', 'Stories', 'Nr_of_rms', 'Units', 'Bdrms', 'Fbath', 'Hbath', 'Lotsize', 'Sale_price'
    ]

    df_suitable = df_encoded[features]

    # Step 8: Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy='mean')  # Use mean for numeric columns
    df_suitable_imputed = pd.DataFrame(imputer.fit_transform(df_suitable), columns=df_suitable.columns)

    # Step 9: Clustering using KMeans (4 clusters for illustration)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_suitable_imputed['cluster'] = kmeans.fit_predict(df_suitable_imputed)

    # Visualizing the clusters with labeled colors
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=df_suitable_imputed['Fin_sqft'], y=df_suitable_imputed['Sale_price'], hue=df_suitable_imputed['cluster'], palette='Set1')
    plt.title('Clustered Residential Data by Property Area and Price')
    plt.xlabel('Property Area (sqft)')
    plt.ylabel('Property Price')
    plt.legend(title='Cluster', labels=['Low Price & Small Area', 'Low Price & Large Area', 'High Price & Small Area', 'High Price & Large Area'])
    plt.show()

    # Save the results (both classification and clustering results)
    df_imputed = df_residential.copy()
    df_imputed['price_class'] = df_residential['price_class']
    df_imputed.to_csv('residential_property_sales_with_classification_and_clustering.csv', index=False)

else:
    print("Dataset was not loaded successfully.")