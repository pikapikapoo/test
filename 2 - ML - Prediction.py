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
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-whitegrid')

# read in data
nlsy=pd.read_csv('https://github.com/Mixtape-Sessions/Machine-Learning/blob/' 
                + 'main/Labs/data/nlsy97.csv?raw=true')
print(nlsy)

#------------------------------------------------------------------------------
# 1. Least squares benchmark
#------------------------------------------------------------------------------
## Generate the regression class for prediction
# generate dictionary of transformations of education
powerlist=[nlsy['educ']**j for j in np.arange(1,10)] # This is a list of lists!
X=pd.concat(powerlist,axis=1) # Make it a dataframe
X.columns = ['educ'+str(j) for j in np.arange(1,10)] # Rename columns

# X_scaled: Covariates for training the model
'''
Standardize our X matrix
- It doesn't matter for OLS, but will matter for lasso below
'''
scaler = StandardScaler() # Generate scaler object
scaler.fit(X)
X_scaled = scaler.transform(X)

# run least squares regression
# instantiate and fite our regression object:
reg=linear_model.LinearRegression().fit(X_scaled,nlsy['lnw_2016'])
# generate predicted values
yhat=reg.predict(X_scaled) # It is not used at all

## plot predicted values
# Actual output
lnwbar=nlsy.groupby('educ')['lnw_2016'].mean()
# Xbar: Covariates for prediction
Xbar=pd.DataFrame({'educ':lnwbar.index.values})
powerlist=[Xbar['educ']**j for j in np.arange(1,10)]
Xbar=pd.concat(powerlist,axis=1)
Xbar.columns = ['educ'+str(j) for j in np.arange(1,10)]
Xbar_scaled = scaler.transform(Xbar) # Standardize using the previously generated one
# Predicted output
ybarhat=reg.predict(Xbar_scaled)
# Plot
fig = plt.figure()
ax = plt.axes()
ax.plot(Xbar['educ1'],lnwbar,'bo',Xbar['educ1'],ybarhat,'g-')
plt.title("ln Wages by Education in the NLSY")
plt.xlabel("years of schooling")
plt.ylabel("ln wages")

#------------------------------------------------------------------------------
# 2. Lasso in action - a single underlying regressor: education
#------------------------------------------------------------------------------
# fit lasso with a couple of different alphas and plot results
# instantiate and fit our lasso object
lasso1 = linear_model.Lasso(alpha=.001,max_iter=1000).fit(X_scaled,nlsy['lnw_2016'])
# generate predicted values
ybarhat1=lasso1.predict(Xbar_scaled)

# same thing but with a different alpha
lasso2 = linear_model.Lasso(alpha=.01,max_iter=1000).fit(X_scaled,nlsy['lnw_2016'])
ybarhat2=lasso2.predict(Xbar_scaled)

# plot
fig1,(ax11,ax12,ax13) = plt.subplots(1,3,figsize=(12, 4))
ax11.barh(Xbar.columns,reg.coef_,align='center');
ax11.set_title("OLS coefficients")
ax11.set_xlabel("coefficient")
ax12.barh(Xbar.columns,lasso1.coef_,align='center');
ax12.set_title("Lasso coefficients (alpha = {:.3f})".format(lasso1.get_params()['alpha']))
ax12.set_xlabel("coefficient")
ax13.barh(Xbar.columns,lasso2.coef_,align='center');
ax13.set_title("Lasso coefficients (alpha = {:.2f})".format(lasso2.get_params()['alpha']))
ax13.set_xlabel("coefficient")

fig2,(ax21,ax22,ax23) = plt.subplots(1,3,figsize=(12,4))
ax21.plot(Xbar['educ1'],lnwbar,'bo',Xbar['educ1'],ybarhat,'g-');
ax21.set_title("ln Wages by Education in the NLSY")
ax21.set_xlabel("years of schooling")
ax21.set_ylabel("ln wages");
ax22.plot(Xbar['educ1'],lnwbar,'bo',Xbar['educ1'],ybarhat1,'g-');
ax22.set_title("ln Wages by Education in the NLSY")
ax22.set_xlabel("years of schooling")
ax22.set_ylabel("ln wages");
ax23.plot(Xbar['educ1'],lnwbar,'bo',Xbar['educ1'],ybarhat2,'g-');
ax23.set_title("ln Wages by Education in the NLSY")
ax23.set_xlabel("years of schooling")
ax23.set_ylabel("ln wages");

#------------------------------------------------------------------------------
# 3. Data-driven tuning parameters: Cross-validation
#------------------------------------------------------------------------------
# define grid for alpha
alpha_grid = {'alpha': [.0001,.001,.002, .004, .006, .008, .01, .012, .014, .016 ,.018, .02 ],
                'max_iter': [100000]}
# instantiate and fit our gridsearchcv object
grid_search = GridSearchCV(
                    linear_model.Lasso(), alpha_grid, cv=5, return_train_score=True
                    ).fit(X_scaled,nlsy['lnw_2016'])
# print out the chosen value for alpha
print("Best alpha: ",grid_search.best_estimator_.get_params()['alpha'])

#------------------------------------------------------------------------------
# 4. Lasso-guided variable selection - select among a large # of regressors
#------------------------------------------------------------------------------
'''
So far we have used only 9 polynomials of educ.
Now we are using the original features in the dataset.
'''
# Define "menu" of regressors:
X=nlsy.drop(columns=['lnw_2016','exp']) # We already have exp as dummies

# Divide into training and test set so we can honestly gauge predictive accuracy
X_train, X_test, y_train, y_test = train_test_split(X, nlsy['lnw_2016'],random_state=42)
# Scale regressors
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Do cross-validated Lasso (the easy way!)
# instantiate and fit our lassocv object
lassocv=linear_model.LassoCV(
                        random_state=42
                        ).fit(X_train_scaled,y_train) 
                        # LassoCV does the long lines of GridSearchCV
# print out the chosen value for alpha
print("Chosen alpha: {:.3f}".format(lassocv.alpha_))
# print the original number of regressors and the number selected by lasso
print("Number of regressors in the menu: ",len(X.columns))
print("Number of regressors selected by lasso: ",sum(lassocv.coef_!=0))
# print out accuracy on training and test test
print("Accuracy on training set: {:.3f}".format(lassocv.score(X_train_scaled,y_train)))
print("Accuracy on test set: {:.3f}".format(lassocv.score(X_test_scaled,y_test)))
# look at the coefficients
results = pd.DataFrame({'feature': X.columns[lassocv.coef_!=0],
                        'coefficient': lassocv.coef_[lassocv.coef_!=0]})
print(results)

#------------------------------------------------------------------------------
# 5. Ridge regression
#------------------------------------------------------------------------------
ridgecv=linear_model.RidgeCV(
                        cv=5,alphas=(.1,1,10,50,100,1000)
                        ).fit(X_train_scaled,y_train)
print("Chosen alpha: {:.3f}".format(ridgecv.alpha_))
print("Accuracy on training set: {:.3f}".format(ridgecv.score(X_train_scaled,y_train)))
print("Accuracy on test set: {:.3f}".format(ridgecv.score(X_test_scaled,y_test)))
# look at the coefficients
results = pd.DataFrame({'feature': X.columns[ridgecv.coef_!=0],
                        'coefficient': ridgecv.coef_[ridgecv.coef_!=0]})
print(results)

# plot
plt.plot(ridgecv.coef_, 's', label="RidgeCV")
plt.plot(lassocv.coef_, 'o', label="LassoCV")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.legend()

#------------------------------------------------------------------------------
# 6. Elastic Net: best of both worlds?
#------------------------------------------------------------------------------
# We don't need to provide 'alpha', which is set automatically
# l1_ratio = gamma
# It takes 13 mins to run the Elastic Net model...
encv=linear_model.ElasticNetCV(
                random_state=42, l1_ratio=[.1, .5, .7, .9, .95, .99, 1]
                ).fit(X_train_scaled,y_train)
print("Chosen l1 ratio: {:.3f}".format(encv.l1_ratio_))
print("Chosen alpha: {:.3f}".format(encv.alpha_))
print("Number of regressors in the menu: ",len(X.columns))
print("Number of regressors selected by elastic net: ",sum(encv.coef_!=0))
print("Accuracy on training set: {:.3f}".format(
                            encv.score(X_train_scaled,y_train)))
print("Accuracy on test set: {:.3f}".format(
                            encv.score(X_test_scaled,y_test)))
# look at the coefficients
results = pd.DataFrame({'feature': X.columns[encv.coef_!=0],
                        'coefficient': encv.coef_[encv.coef_!=0]})
print(results)

#------------------------------------------------------------------------------
# 7. Decision Trees and Random Forests
#------------------------------------------------------------------------------
## Import some utilities
# Run the following code only if we run it for the first time.
'''
#@title
import requests
url1 = 'https://www.dropbox.com/s/jgml061manxpawo/plot_2d_separator.py?raw=true'
url2 = 'https://www.dropbox.com/s/hlrrlwm4kt36awb/plot_interactive_tree.py?raw=true'
url3 = 'https://www.dropbox.com/s/e2cy203sr30a59z/plot_helpers.py?raw=true'
url4 = 'https://www.dropbox.com/s/aik5sgcwgz4brwn/tools.py?raw=true'
r1 = requests.get(url1)
r2 = requests.get(url2)
r3 = requests.get(url3)
r4 = requests.get(url4)

# make sure your filename is the same as how you want to import 
with open('plot_2d_separator.py', 'w') as f1:
    f1.write(r1.text)

with open('plot_interactive_tree.py', 'w') as f2:
    f2.write(r2.text)

with open('plot_helpers.py', 'w') as f3:
    f3.write(r3.text)

with open('tools.py', 'w') as f4:
    f4.write(r4.text)
'''

# now we can import
import plot_helpers
import tools
import plot_2d_separator
import plot_interactive_tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz # I used this package for DAGs!
from sklearn.datasets import make_moons

## Generate a toy dataset
'''
- sklearn.datasets: Make toy dataset to visualize clustering 
                and classification algorithms.
- make_moons: Make two interleaving half circles.
'''
Xfake, yfake = make_moons(n_samples=100, noise=0.25, random_state=3)
# Visualize the fake dataset
plt.title("Crescent-shaped clusterring") # White (black) circle if y=0 (1)
plt.scatter(Xfake[:, 0], Xfake[:, 1], marker='o', c=yfake, s=100,
            edgecolor="k", linewidth=2)
plt.xlabel("$Xfake_1$")
plt.ylabel("$Xfake_2$")
plt.show()

## Train dataset vs. test dataset
(Xfake_train, Xfake_test, 
yfake_train, yfake_test) = train_test_split(Xfake, yfake,
                                            stratify=yfake, random_state=42)

## First a simple tree:
tree = DecisionTreeClassifier(max_depth=3).fit(Xfake_train, yfake_train)
fig1,ax = plt.subplots(1,1,figsize=(12, 8))
plot_interactive_tree.plot_tree_partition(Xfake_train, yfake_train, tree, ax=ax)
dot_data= export_graphviz(tree, out_file=None, impurity=False, filled=True)
graph = graphviz.Source(dot_data) 
graph

## Now average over several trees:
forest = RandomForestClassifier(n_estimators=5, random_state=2
                                    ).fit(Xfake_train, yfake_train)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    plot_interactive_tree.plot_tree_partition(
                                Xfake_train, yfake_train, tree, ax=ax)

plot_2d_separator.plot_2d_separator(
                                forest, Xfake_train,
                                fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
plot_helpers.discrete_scatter(Xfake_train[:, 0], Xfake_train[:, 1], yfake_train)


#------------------------------------------------------------------------------
# 8. Predict wages in the NLSY by random forests, 
# just as we did for Lasso, Ridge, and Elastic net.
#------------------------------------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

# First without cross-validating
rf=RandomForestRegressor(random_state=42).fit(X_train,y_train)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test,y_test)))

# Now with cross-validation - There is no package for CV
# define grid for max_depth
param_grid = {'max_depth': [5,10,100]}
grid_searchrf = GridSearchCV(
                        RandomForestRegressor(),
                        param_grid,
                        cv=5,
                        return_train_score=True
                        ).fit(X_train,y_train)
print("Best max_depth: ",grid_searchrf.best_estimator_.get_params()['max_depth'])
print("Accuracy on training set: {:.3f}".format(
                                        grid_searchrf.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(
                                        grid_searchrf.score(X_test,y_test)))
'''
We get the similar results as LASSO.

Advantage of Random Forest
1. Independent of the scale of X; no need to standardize covariates.
2. CV is extremely valid; it is less sensitive to tuning parameters 
(not so sensitive to depth of tree)
'''

# If you want to plot the tree 1 among 100 trees.
dot_data= export_graphviz(
                rf.estimators_[0], out_file=None, impurity=False, filled=True)
graph = graphviz.Source (dot_data)
graph

