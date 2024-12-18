# Patient Survival Prediction API

This API predicts the survival probability of a patient during hospitalization based on various health and lifestyle metrics. It uses a trained machine learning model to compute the probability of survival.

---

## Install the Required Python Packages

Install the necessary dependencies to run the application:

```bash
pip install fastapi uvicorn scikit-learn joblib pandas numpy kagglehub imbalanced-learn
```

---

## Run the Application

Start the FastAPI application:

```bash
python3 train_model.py  
uvicorn app:app --reload
```

---

## Make a Request

To interact with the API, send a POST request to the `/predict` endpoint. Here's the structure of the request and an explanation of the fields.

### Request Structure
Send the following JSON data in the body of your request:

```json
{
    "features": {
        "age": 56.0,
        "gender": "Male",
        "bmi": 22.1,
        "ethnicity": "Caucasian",
        "apache_4a_hospital_death_prob": 0.05,
        "solid_tumor_with_metastasis": 0.0
    }
}
```

### Explanation of Fields

1. **age**: Age of the patient (float).
2. **gender**: Gender of the patient. Accepted values are:
   - "Male"
   - "Female"
3. **bmi**: Body Mass Index (float).
4. **ethnicity**: Ethnic background. Accepted values include:
   - "Caucasian"
   - "African American"
   - "Asian"
   - "Hispanic"
   - ... (other encoded values).
5. **apache_4a_hospital_death_prob**: Predicted hospital death probability from the APACHE IV scoring system (float).
6. **solid_tumor_with_metastasis**: Indicator for the presence of a solid tumor with metastasis (0.0 or 1.0).

---

## Example curl Command

Here’s an example of how to send a request to the API:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "features": {
        "age": 56.0,
        "gender": "Male",
        "bmi": 22.1,
        "ethnicity": "Caucasian",
        "apache_4a_hospital_death_prob": 0.05,
        "solid_tumor_with_metastasis": 0.0
    }
}'
```

---

## Response Structure

The API will return a JSON object containing the survival probability:

```json
{"survival_probability":0.41749318577227157}
```

### Response Fields
- **survival_probability**: A float value representing the likelihood of survival during hospitalization.

---

## Steps to Build the Project

### 1. Data Collection and Preprocessing
- **Dataset**: The dataset was sourced from Kaggle and contains patient health metrics and outcomes.
- **Preprocessing Steps**:
  - Missing values were imputed using the mean for numerical data and the most frequent value for categorical data.
  - Categorical variables were encoded using `LabelEncoder`.
  - Numerical data was standardized using `StandardScaler`.

### 2. Model Selection and Training
- **Model**: A `RandomForestClassifier` was selected for its ability to handle high-dimensional data.
- **Data Balancing**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the target variable.
- **Hyperparameters**:
  - Number of estimators: 50
  - Maximum depth: 10
- **Train-Test Split**: Data was split into training and testing sets with an 80%-20% ratio.

### 3. Model Evaluation
- The model’s performance was evaluated using the following metrics:
  - Accuracy
  - ROC-AUC score

### 4. API Development
- **Framework**: FastAPI was used to create the API for serving predictions.
- **Preprocessing Integration**:
  - Encoded categorical variables using saved `LabelEncoder` artifacts.
  - Scaled numerical variables using the saved `StandardScaler`.
- **Endpoint**: The `/predict` endpoint accepts patient data in JSON format and returns the survival probability.

### 5. Deployment
- The trained model and preprocessing artifacts were saved using `joblib` for efficient reloading.
- The API can be hosted locally using Uvicorn or deployed to a cloud platform for broader accessibility.

