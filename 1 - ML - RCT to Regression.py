'''
Wonjin Lee
10-29-2022
-------------------------------------------------------------------------------
Mixtape - ML by Brigham Frandsen

Read the Jupyter notebook!
-------------------------------------------------------------------------------
'''
# import some useful packages
import pandas as pd
# import numpy as np
from sklearn import linear_model

# read in data
oregonhie=pd.read_csv('https://github.com/Mixtape-Sessions/Machine-Learning/'
                    +'blob/main/Labs/data/oregon_hie_table5.csv?raw=true')

#------------------------------------------------------------------------------
# 1. Gold standard: RCT
#------------------------------------------------------------------------------
''''
data cleaning: drop observations with missing values in any of the variables 
we are going to need: 
'''
regvarnames=(['doc_num','treatment','weight']
                +[col for col in oregonhie if col.startswith('ddd')])
regvars=oregonhie[regvarnames].dropna()

# define outcome, treatment, weights, additional covariates
y=regvars['doc_num'] # This is a series.
d=regvars[['treatment']] # This is a data frame.
w=regvars['weight']
x=regvars.drop(['doc_num','treatment','weight'],axis=1)
print(regvars)

# run weighted regression of outcome on treatment
lm=linear_model.LinearRegression()
lm.fit(d,y,w)

# display treatment effect
print("Estimated effect of Medicaid elibility on \n number of doctor visits" +
    " (bivariate): {:.3f}".format(lm.coef_[0]))


#------------------------------------------------------------------------------
# 2. Aluminum standard: Regression control
#------------------------------------------------------------------------------
# Add the household size indicators to our regressor set and run regression:
lm.fit(pd.concat([d,x],axis=1),y,w)
print("Estimated effect of Medicaid elibility on \n number of doctor visits" +
    " (with controls): {:.3f}".format(lm.coef_[0]))


#------------------------------------------------------------------------------
# 3. Connection to ML
#------------------------------------------------------------------------------
# Regress outcome on covariates
yreg=linear_model.LinearRegression().fit(x,y,w)
# Calculate residuals
ytilde = y - yreg.predict(x)

# regress treatment on covariates
dreg = linear_model.LinearRegression().fit(x,d,w)
# Calculate residuals
dtilde = d - dreg.predict(x)

# regress ytilde on dtilde
lm.fit(dtilde,ytilde,w)
print("Estimated effect of Medicaid elibility on \n number of doctor visits" +
    " (partialled out): {:.3f}".format(lm.coef_[0]))