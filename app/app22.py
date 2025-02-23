import pandas as pd
import requests
from io import BytesIO
import pickle

# The raw URL of your .pkl file
url = 'https://raw.githubusercontent.com/dBCooper2/hacklytics-nhl-dashboard/blob/main/pkl_files/logistic_model.pkl'

# Fetch the file content using requests
response = requests.get(url)

# Ensure the request was successful
if response.status_code == 200:
    # Use BytesIO to load the pickle file from the response
    pickle_file = BytesIO(response.content)
    
    # Now you can load the dataframe using pd.read_pickle
    df = pickle.load(pickle_file, 'rb')
    print(df.head())  # Example to see the dataframe content
else:
    print("Failed to retrieve the file.")
