import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import plotly.graph_objects as go

# get_prepped_data()
#  - Accesses Data from Google Drive and Preps it for Models
#  - Returns a Test-Train Split as a dictionary with key-value pairs along with the original DataFrame for plotting:
# { 
#   "wp_df" : wp_df
#   "x-test" : X_test,
#   "y-test" : Y_test,
#   "x-train" : X_train,
#   "y-train" : Y_train,
#   "x-train-res" : X_train_res,
#   "y-train-res" : Y_train_res,
# }
def get_prepped_data()->dict:
    # Data Paths :: CHANGE TO GOOGLE DRIVE FOLDER!!!
    pbp_path = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/.data/nhl_pbp20222023.csv'
    shifts_path = '/Users/dB/Documents/repos/github/hacklytics-nhl-dashboard/.data/nhl_shifts20222023.csv'

    # Read in Data ::
    pbp = pd.read_csv(pbp_path)
    shifts = pd.read_csv(shifts_path)

    # Preparing Data ::
    pbp.groupby(['Game_Id', 'Ev_Team'])['Game_Id'].count()

    # Calculate Max Goals for Home and Away Teams for Binary Classifiers for W/L
    pbp['home_max_goal'] = pbp['Home_Score'].groupby(pbp['Game_Id']).transform('max')
    pbp['away_max_goal'] = pbp['Away_Score'].groupby(pbp['Game_Id']).transform('max')

    # WP Model DataFrame
    pbp['home_max_goal'] = pbp.groupby('Game_Id')['Home_Score'].transform('max')
    pbp['away_max_goal'] = pbp.groupby('Game_Id')['Away_Score'].transform('max')

    # Calculate Win Observation
    pbp['win'] = np.where(pbp['home_max_goal'] > pbp['away_max_goal'], 1, 0)

    # Fix Time
    pbp['total_seconds_elapsed'] = ((pbp['Period'] - 1) * 1200) + pbp["Seconds_Elapsed"]
    pbp['time_remaining'] = 3600 - pbp['total_seconds_elapsed']

    pbp['score_diff'] = pbp['Home_Score'] - pbp['Away_Score'] # Compute Score Differential

    pbp['skater_diff'] = pbp['Home_Players'] - pbp['Away_Players']
    pbp['goalie_pulled'] = np.where(pbp['Home_Players'] < 5, 1, 0)
    pbp['power_play'] = np.where((pbp['Home_Players'] > pbp['Away_Players']) | (pbp['Away_Players'] > pbp['Home_Players']), 1, 0)

    shifts_agg = shifts.groupby(['Game_Id', 'Player_Id'])['Duration'].sum().reset_index()
    pbp = pbp.merge(shifts_agg, left_on=['Game_Id', 'p1_ID'], right_on=['Game_Id', 'Player_Id'], how='left')
    pbp.rename(columns={'Duration': 'ice_time'}, inplace=True)
    pbp['ice_time'] = pbp['ice_time'].fillna(0)

    wp_df = pbp[['Game_Id', 'p1_name', 'Ev_Team', 'time_remaining', 'score_diff', 'skater_diff', 'goalie_pulled', 'ice_time', 'win']]
    wp_df = wp_df.dropna()

    # WP Model :: Test-Train Split
    train_games, test_games = train_test_split(wp_df['Game_Id'].unique(), test_size=0.3, random_state=2142)
    train_df = wp_df[wp_df['Game_Id'].isin(train_games)]
    test_df = wp_df[wp_df['Game_Id'].isin(test_games)]


    X_train = train_df.drop(['win', 'p1_name', 'Ev_Team', 'Game_Id'], axis=1)
    Y_train = train_df[['win']]

    X_test = test_df.drop(['win', 'p1_name', 'Ev_Team', 'Game_Id'], axis=1)
    Y_test = test_df[['win']]


    sm = SMOTE(random_state=2142)
    X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train)

    res = {
        'pbp_df':pbp,
        'wp_df':wp_df,
        'x-test': X_test,
        'y-test': Y_test,
        'x-train': X_train,
        'y-train': Y_train,
        'x-train-res': X_train_res,
        'y-train-res': Y_train_res,
    }

    return res

# lr_model(tt_split)

def get_lr_model(tt_split: dict):

    x_test = tt_split['x-test']
    x_train_res = tt_split['x-train-res']
    y_train_res = tt_split['y-train-res']

    model = LogisticRegression()
    model.fit(x_train_res, y_train_res.values.ravel())
    Y_pred = model.predict(x_test)
    Y_pred_proba = model.predict_proba(x_test)[:, 1]

    res = {
            'model':model,
            'y-pred':Y_pred,
            'y-pred-probs':Y_pred_proba,
          }
    
    return res

def get_lr_metrics(tt_split: dict, lr_model:dict)->dict:

    Y_test = tt_split['y-test']
    Y_pred = lr_model['y-pred']
    Y_pred_proba = lr_model['y-pred-probs']

    # Logistic Regression :: Accuracy
    lr_accuracy = accuracy_score(Y_test, Y_pred)
    lr_precision = precision_score(Y_test, Y_pred)
    lr_recall = recall_score(Y_test, Y_pred)
    lr_roc_auc = roc_auc_score(Y_test, Y_pred_proba)

    res = {
        'lr_accuracy':lr_accuracy,
        'lr_precision':lr_precision,
        'lr_recall':lr_recall,
        'lr_roc_auc':lr_roc_auc,
    }

    return res

'''
# Logistic Regression :: ROC Plot and AUC
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {lr_roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Logistic Regression :: Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Loss (0)", "Win (1)"], yticklabels=["Loss (0)", "Win (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
'''