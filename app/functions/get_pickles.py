import lr_model as lr
import xgb_model as xgb
import pickle
import pandas as pd

from joblib import load

x = lr.get_prepped_data()

lr_m = lr.get_lr_model(x)['lr-model']
xgb_m = xgb.get_xgboost_model(x)['xgb-model']

print(type(lr_m))
print(type(xgb_m))

lr_filename = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/pkl_files/logistic_model.pkl'
xgb_filename = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/pkl_files/xgboost_model.pkl'

pickle.dump(lr_m, open(lr_filename, 'wb'))
pickle.dump(xgb_m, open(xgb_filename, 'wb'))

nu_lr = pickle.load(open(lr_filename, 'rb'))
nu_xgb = pickle.load(open(xgb_filename, 'rb'))

print(type(nu_lr)==type(lr_m))
print(type(nu_xgb)==type(xgb_m))

import gdown

file_id = "11BMLszgKfN1_DFucS6Ihzlz4b6J3IdMT"
url = f"https://drive.google.com/uc?id={file_id}"
gdown.download(url, "logistic_model.pkl", quiet=False)

# Now load the pickle file
import pickle
with open("logistic_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

print(type(lr_model))