import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
csv_filename = 'Sleep_health_and_lifestyle_dataset.csv'
file_path = os.path.join(path, csv_filename)

if os.path.exists(file_path):
    df = pd.read_csv(file_path)

df = df.ffill()

df = df[[
    'Gender', 'Age', 'Sleep Duration', 'Physical Activity Level', 
    'Stress Level', 'Sleep Disorder'
]].rename(columns={
    'Sleep Duration': 'SleepDuration',
    'Physical Activity Level': 'PhysicalActivityLevel',
    'Stress Level': 'StressLevel',
    'Sleep Disorder': 'SleepDisorder'
})

df = df[df['SleepDisorder'].notna()]

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['SleepDisorder'] = df['SleepDisorder'].map({
    'No Sleep Disorder': 0, 'Sleep Apnea': 1, 'Insomnia': 2
})

thresholds = {}
for disorder in df['SleepDisorder'].unique():
    disorder_data = df[df['SleepDisorder'] == disorder]
    thresholds[disorder] = {
        "SleepDuration": (
            disorder_data['SleepDuration'].mean() - disorder_data['SleepDuration'].std(),
            disorder_data['SleepDuration'].mean() + disorder_data['SleepDuration'].std()
        ) if not disorder_data['SleepDuration'].empty else (7, 9),
        "PhysicalActivityLevel": (
            disorder_data['PhysicalActivityLevel'].quantile(0.25),
            disorder_data['PhysicalActivityLevel'].quantile(0.75)
        ) if not disorder_data['PhysicalActivityLevel'].empty else (60, 100),
        "StressLevel": (
            disorder_data['StressLevel'].quantile(0.25),
            disorder_data['StressLevel'].quantile(0.75)
        ) if not disorder_data['StressLevel'].empty else (0, 3)
    }

if 0 not in thresholds:
    thresholds[0] = {
        "SleepDuration": (7, 9),
        "PhysicalActivityLevel": (60, 100),
        "StressLevel": (0, 3)
    }

X = df.drop('SleepDisorder', axis=1)
y = df['SleepDisorder']

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
model = grid_search.best_estimator_

y_pred = model.predict(X_test_scaled)

class_index_to_name = {0: "No Sleep Disorder", 1: "Sleep Apnea", 2: "Insomnia"}
unique_classes = np.unique(y_test)

dynamic_target_names = [class_index_to_name[class_idx] for class_idx in unique_classes]

print("Model Performance on Test Data:")
print(classification_report(y_test, y_pred, target_names=dynamic_target_names, labels=unique_classes))

feature_importances = model.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

joblib.dump(model, 'sleep_disorder_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

app = FastAPI()

model = joblib.load('sleep_disorder_model.pkl')
scaler = joblib.load('scaler.pkl')

class SleepData(BaseModel):
    Gender: str
    Age: int
    SleepDuration: float
    PhysicalActivityLevel: float
    StressLevel: float

@app.post("/predict")
def predict(data: SleepData):
    input_data = pd.DataFrame([[
        1 if data.Gender.lower() == 'male' else 0,
        max(data.Age, 27),
        data.SleepDuration,
        data.PhysicalActivityLevel,
        data.StressLevel
    ]], columns=X.columns)

    scaled_data = scaler.transform(input_data)
    probabilities = model.predict_proba(scaled_data)[0]

    num_classes = len(probabilities)

    if num_classes == 3:
        if probabilities[0] > 0.6:
            prediction_label = "No Sleep Disorder"
        elif probabilities[1] > 0.6:
            prediction_label = "Sleep Apnea"
        elif probabilities[2] > 0.7:
            prediction_label = "Insomnia"
        else:
            if data.SleepDuration <= 6.5 and data.StressLevel >= 7:
                prediction_label = "Insomnia"
            else:
                scores = {}
                for disorder, rules in thresholds.items():
                    score = 0
                    if rules["SleepDuration"][0] <= data.SleepDuration <= rules["SleepDuration"][1]:
                        score += 0.5
                    if rules["PhysicalActivityLevel"][0] <= data.PhysicalActivityLevel <= rules["PhysicalActivityLevel"][1]:
                        score += 0.3
                    if rules["StressLevel"][0] <= data.StressLevel <= rules["StressLevel"][1]:
                        score += 0.2
                    scores[disorder] = score

                best_fit_disorder = max(scores, key=scores.get)
                prediction_label = class_index_to_name[best_fit_disorder]

    elif num_classes == 2:
        if probabilities[0] > 0.6:
            prediction_label = "No Sleep Disorder"
        else:
            prediction_label = "Sleep Apnea"
    else:
        prediction_label = "Uncertain"

    description_map = {
        "No Sleep Disorder": {
            "description": "No sleep disorder detected. This means the individual is experiencing normal sleep patterns.",
            "suggestion": "Maintain a healthy lifestyle with regular sleep schedules, good diet, and physical activity."
        },
        "Sleep Apnea": {
            "description": "Sleep Apnea is a condition where breathing repeatedly stops and starts during sleep.",
            "suggestion": "Consult a healthcare provider for sleep studies and consider using CPAP therapy."
        },
        "Insomnia": {
            "description": "Insomnia is a sleep disorder characterized by difficulty falling or staying asleep.",
            "suggestion": "Improve sleep hygiene by avoiding caffeine and electronic screens before bed."
        }
    }

    description = description_map.get(prediction_label, {"description": "No description available.", "suggestion": "Consult a healthcare provider."})

    return {
        "prediction": prediction_label,
        "description": description["description"],
        "suggestion": description["suggestion"],
        "probabilities": probabilities.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
