import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# read data from csv
def read_data():
    train = pd.read_csv('../input/train_features')
    test = pd.read_csv('../input/test_features')
    label = pd.read_csv('../input/train_label',header=None)
    return train,test,label

# C:penalty parameter
# loss:loss function
# epsilon:Epsilon parameter in the epsilon-insensitive loss function
# dual:determine either solve the dual or primal optimization problem,Prefer dual=False when n_samples > n_features
# tol: Tolerance for stopping criteria
clf = svm.LinearSVR()

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
def model_test():
    X_train,X_test,y = read_data()
    # cross-validation: aviod overfitting
    # step1: split trainning data
    #cro_train, cro_test, cro_y_train, cor_y_test = train_test_split(X_train, y, test_size=0.4, random_state=0)
    #clf.fit(cro_train,np.array(cro_y_train))
    #clf.score(cro_test,np.array(cro_y_test))
    print(cross_val_score(clf, X_train, np.array(y),cv=5,scoring='neg_mean_squared_error'))

def model_use():
    X_train,X_test,y = read_data()
    clf.fit(X_train,y)
    pre_val = np.expm1(pd.DataFrame(clf.predict(X_test)))
    test_label = pd.read_csv('../input/test_id',header=None)
    result = pd.DataFrame()
    result['Id'] = test_label[0]
    result['SalePrice'] = pre_val[0]
    result.to_csv('../output/result_svr',index=None)

model_use()
        
   
#model_test() 
