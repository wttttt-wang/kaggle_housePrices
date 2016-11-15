import xgboost as xgb
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV

# read data from csv
def read_data():
    train = pd.read_csv('train_features')
    test = pd.read_csv('test_features')
    label = pd.read_csv('train_label',header=None)
    return train,test,label

# xgboost model
def xgb_model():
    X_train,X_test,y = read_data()
    dtrain = xgb.DMatrix(X_train, label = y)
    dtest = xgb.DMatrix(X_test)
    ''' #using xgb.cv to tuned the params
    params = {"max_depth":2, "eta":0.1}
    model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
    model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
    plt.savefig('rmse-xgboost.png')
    '''
    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
    model_xgb.fit(X_train, y)   # fit the model
    ''' # output after fitting
    XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
       learning_rate=0.1, max_delta_step=0, max_depth=2,
       min_child_weight=1, missing=None, n_estimators=360, nthread=-1,
       objective='reg:linear', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)
    '''
    xgb_preds = np.expm1(model_xgb.predict(X_test)) # the prediction value of xgboost
    model_lasso = LassoCV(alphas = [1,0.1,0.001,0.0005]).fit(X_train,y) 
    lasso_preds = np.expm1(model_lasso.predict(X_test)) # the prediction value of lasso
    predictions = pd.DataFrame({'xgb':xgb_preds,'lasso':lasso_preds})
    predictions.plot(x = 'xgb',y = 'lasso',kind='scatter')
    plt.savefig('prediction_xgb_lasso)')

xgb_model()
