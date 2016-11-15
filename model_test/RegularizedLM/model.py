import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# read the handled data
X_train=pd.read_csv('train_features') 
X_test = pd.read_csv('test_features') 
y = pd.read_csv('train_label',header=None) # actually i didnt store the header for the labels

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model,X_train,y,scoring='neg_mean_squared_error',cv=5))
    return(rmse)

def model_selection():
    model_ridge = Ridge()
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]
    cv_ridge = pd.Series(cv_ridge,index=alphas)
    print 'rmse of ridge',
    print cv_ridge
    ''' # png for rmse-alpha(ridge)
    cv_ridge.plot(title = "Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")
    plt.savefig('ridge_alpha.png',dpi=150)
    '''
    # lassoCV can figure out the best alpha for us
    model_lasso = LassoCV(alphas = [1,0.1,0.001,0.0005]).fit(X_train,y)
    print 'rmse of lasso',
    print rmse_cv(model_lasso).mean()
# model_selection()

# after the comparison upside
# we find that the Lasso performs better, so we use it here to predict on the test set
# one of the neat thing of Lasso is that it does feature selection for us, by
# setting coeffcients of features that is deems unimportant to zero
# Just take a look at the coefficients
model_lasso = LassoCV(alphas = [1,0.1,0.001,0.0005]).fit(X_train,y)
def coef_lasso():
    coef = pd.Series(model_lasso.coef_,index = X_train.columns)
    print 'cofficients of lasso',
    print coef
    # count that how many features Lasso picked
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# But note that here,features selected are not necessarily the 'correct' ones especially since here are a lot of collinear features in the dataset.
# One idea to try here is run Lasso a few times on boostrapped samples and see how stable the feature selection is.

    '''
    # First ,take a direct look at what the most important coefficients are, by a png.
    imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.savefig('coef in Lasso.png')
    '''
def residual_lasso():
    # let's look at the residuals as well:
    matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
    #preds = pd.DataFrame({"preds":model_lasso.predict(X_train),"true":y})
    preds = pd.DataFrame({"preds":model_lasso.predict(X_train)})
    preds['true'] = y
    preds["residuals"] = preds['true']-preds['preds']
    preds.plot(x = "preds", y = "residuals",kind = "scatter")
    plt.savefig('residual_Lasso.png')

#residual_lasso()
