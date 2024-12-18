import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

# Download and load the dataset
path = kagglehub.dataset_download("gbiamgaurav/patient-survival-prediction")
dataset_path = f"{path}/Dataset.csv"

# Load dataset
data = pd.read_csv(dataset_path)

# Preprocessing
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
target_column = 'hospital_death'
numerical_columns = numerical_columns.difference([target_column])
categorical_columns = data.select_dtypes(include=['object']).columns

# Impute missing values
numerical_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Scale numerical columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Save preprocessing artifacts
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(list(data.columns), 'dataset_columns.pkl')

# Prepare target and features
y = data[target_column].astype(int)
X = data.drop(target_column, axis=1)

# Save feature names for reference
print("Feature names used for training:")
print(list(X.columns))
with open("feature_names.txt", "w") as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

# Balance the classes using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)

# Train the model
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'survival_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model, scaler, label encoders, and dataset columns have been saved.")
