import numpy as np
import pandas as pd
import xgboost as xgb

# read data from csv
def read_data():
    train = pd.read_csv('../input/train_features')
    test = pd.read_csv('../input/test_features')
    label = pd.read_csv('../input/train_label',header=None)
    return train,test,label

X_train,X_test,y = read_data()
dtrain = xgb.DMatrix(X_train, label = y)
#dtest = xgb.DMatrix(X_test)

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)
model_xgb.fit(X_train, y)
pre_val = np.expm1(pd.DataFrame(model_xgb.predict(X_test)))
test_label = pd.read_csv('../input/test_id',header=None)
result = pd.DataFrame()
result['Id'] = test_label[0]
result['SalePrice'] = pre_val[0]
result.to_csv('../output/result_xgb.csv',index=None)
