'''
Wonjin Lee
10-30-2022
-------------------------------------------------------------------------------
Mixtape - ML by Brigham Frandsen

Read the Jupyter notebook!
-------------------------------------------------------------------------------
'''
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold

nlsy=pd.read_csv('https://github.com/Mixtape-Sessions/Machine-Learning/'
                    + 'blob/main/Labs/data/nlsy97.csv?raw=true')
print(nlsy)

# Define outcome, regressor of interest
y=nlsy['lnw_2016'] # Dependent variable should be panda's Series.
d=nlsy[['black']] # Covariates should be panda's DataFrame.
#------------------------------------------------------------------------------
# 1. Post Double Selection Lasso (PDS Lasso)
#------------------------------------------------------------------------------
## Simple Regression with no Controls
# instantiate and fit a linear regression object
lm=linear_model.LinearRegression().fit(d,y)
# print out regression coefficient
print("Simple regression race gap: {:.3f}".format(lm.coef_[0]))
print(f"Simple regression race gap: {round(lm.coef_[0],3)}")

# define RHS, matrix of the d and the controls we want
RHS=nlsy[['black','educ','exp','afqt']]
# run regression
lm.fit(RHS,y)
# print out coefficient
print("Multiple regression-adjusted race gap: {:.3f}".format(lm.coef_[0]))

## Post Double Selection Lasso
# let's define a matrix X with all of our potential controls
X=nlsy.drop(columns=['lnw_2016','black'])

# Step 1: Lasso the outcome on X
lassoy = linear_model.LassoCV(max_iter=1000,normalize=True).fit(X, y)

# Step 2: Lasso the treatment on X
lassod = linear_model.LassoCV(max_iter=1000,normalize=True).fit(X, d)

# Step 3: Form the union of controls
'''
1. lassoy.coef_ is 992 sized np array.
2. X is a 1266 by 992 pd DataFrame.
3. Xunion keeps the columns whose lasso coef are positive in X.
'''
Xunion=X.iloc[:,(lassod.coef_!=0) + (lassoy.coef_!=0)]
print(Xunion.head())

# Concatenate treatment with union of controls and regress y on that and 
# print out estimate
rhs=pd.concat([d,Xunion],axis=1)
fullreg=linear_model.LinearRegression().fit(rhs,y)
print("PDS regression earnings race gap: {:.3f}".format(fullreg.coef_[0]))


#------------------------------------------------------------------------------
# 2. Double-Debiased Machine Learning
#------------------------------------------------------------------------------
## 2.1 For simplicity, we will first do it without sample splitting
# Step 1: Ridge outcome on Xs, get residuals
ridgey = linear_model.RidgeCV(normalize=True).fit(X, y)
yresid=y-ridgey.predict(X)

# Step 2: Ridge treatment on Xs, get residuals
ridged = linear_model.RidgeCV(normalize=True).fit(X, d)
dresid=d-ridged.predict(X)

# Step 3: Regress y resids on d resids and print out estimate
dmlreg=linear_model.LinearRegression().fit(dresid,yresid)
print("DML regression earnings race gap: {:.3f}".format(dmlreg.coef_[0]))

## 2.2 The real thing: with sample splitting
# create our sample splitting "object"
kf = KFold(n_splits=5,shuffle=True,random_state=42)

# apply the splits to our Xs
kf.get_n_splits(X)

# initialize columns for residuals
yresid = y*0
dresid = d*0

# Now loop through each fold
ii=0
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    d_train, d_test = d.iloc[train_index,:], d.iloc[test_index,:]
    
    # Do DML thing
    # Ridge y on training folds:
    ridgey.fit(X_train, y_train)

    # but get residuals in test set
    yresid.iloc[test_index]=y_test-ridgey.predict(X_test)
    
    #Ridge d on training folds
    ridged.fit(X_train, d_train)

    #but get residuals in test set
    dresid.iloc[test_index,:]=d_test-ridged.predict(X_test)

# Regress resids
dmlreg=linear_model.LinearRegression().fit(dresid,yresid)
print("DML regression earnings race gap: {:.3f}".format(dmlreg.coef_[0]))

## If you want standard errors, the use statsmodels package
import statsmodels.api as sm
rhs = sm.add_constant(dresid)
model = sm.OLS(yresid, rhs)
results = model.fit(cov_type='HC3')
print(results.summary())


## 2.3 HW: Now do DML using Random Forest!

