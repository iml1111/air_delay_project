import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc,confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.externals import joblib
SEED = 42
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':1,
                    'n_estimators':4000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                }

def l_proc(df):
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
    df_x = df
    for df_index, val_index in split.split(df_x, df_x['SDT_MM']):
        df1 = df_x.iloc[df_index]
        val1 = df_x.iloc[val_index]
    #학습데이터셋 스플릿하기
    df_y = df1['DLY']
    df_x = df1.drop(['DLY'], axis=1)
    #밸리드데이터셋 스플릿하기
    val_y = val1['DLY']
    val_x = val1.drop(['DLY'], axis=1)
    print("split done...")
    tr_data = lgb.Dataset(df_x, label=df_y)
    vl_data = lgb.Dataset(val_x, label = val_y)  
    estimator = lgb.train(
        lgb_params,
        tr_data,
        valid_sets = [tr_data, vl_data],
        verbose_eval = 200,
    )    
    Y_pred = estimator.predict(val_x)
    print(roc_auc_score(val_y, Y_pred))
    return Y_pred