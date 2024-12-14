# Sleep Health and Lifestyle Prediction API

This API predicts sleep disorders based on health and lifestyle data. It uses a trained machine learning model to classify sleep disorders like insomnia, sleep apnea, and narcolepsy.

# Install the required Python packages

```bash
pip install fastapi uvicorn scikit-learn joblib pandas numpy kagglehub
```

# Run the App
```bash
uvicorn sleep_health_model:app --reload
```

# Make a Request
To interact with the API, send a POST request to the /predict endpoint. Here's the structure of the request and the explanation of each field.

Request Structure
Send the following JSON data in the body of your request:
```
{
    "Gender": "female",
    "Age": 52,
    "SleepDuration": 6.6,
    "PhysicalActivityLevel": 7,
    "StressLevel": 7
}
```
Explanation of Fields:
1. Age: The age of the individual (integer).

2.  Gender: The gender of the individual. Accepted values are:

- "Male"
- "Female"

3. Exercise: Whether the person exercises regularly. Accepted values:

- "Yes"
- "No"

4. AlcoholConsumption: Frequency of alcohol consumption. Accepted values:

- "None"
- "1-2 times/week"
- "3-4 times/week"
- "Daily"

5. Diet: The diet habits of the individual. Accepted values:

- "Healthy"
- "Moderate"
- "Unhealthy"

6. SleepDuration: The number of hours of sleep on average (float).

7. PhysicalActivityLevel: Physical activity level of the individual. Accepted values:

- "Low"
- "Moderate"
- "High"

8. StressLevel: The perceived stress level. Accepted values:

- "Low"
- "Moderate"
- "High"

# Example curl Command:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "Gender": "female",
    "Age": 52,
    "SleepDuration": 6.6,
    "PhysicalActivityLevel": 7,
    "StressLevel": 7
}'
```

# Response Structure

```
{"prediction":"Sleep Apnea",
"description":"Sleep Apnea is a condition where breathing repeatedly stops and starts during sleep.",
"suggestion":"Consult a healthcare provider for sleep studies and consider using CPAP therapy.","probabilities":[0.145,0.855]}
```

