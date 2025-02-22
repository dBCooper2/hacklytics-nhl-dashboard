import joblib
import requests
from io import BytesIO

# The raw URL of your .pkl file (sklearn model)
url = 'https://github.com/dBCooper2/hacklytics-nhl-dashboard/blob/main/pkl_files/logistic_model.pkl'

import pickle
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Fetch the file content using requests
response = requests.get(url)

# Ensure the request was successful
if response.status_code == 200:
    # Use BytesIO to load the pickle file from the response content
    model_file = BytesIO(response.content)
    
    # Load the sklearn model using joblib (or pickle if you used pickle to serialize the model)
    model = joblib.load(model_file)
    
    # Now you can use your model to make predictions or perform other operations
    # Example prediction (assuming the model is a classifier)
    sample_data = [[1.5, 2.3, 3.1]]  # Example input data (replace with actual features)
    prediction = model.predict(sample_data)
    print(prediction)
else:
    print("Failed to retrieve the model.")

    