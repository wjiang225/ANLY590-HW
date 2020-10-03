import sklearn
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

##############Problem 1.1
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999
df = pd.read_csv('Hitters.csv')
df = df.dropna()
df_num = df.select_dtypes(exclude=['string','object']).drop(['Salary'],axis=1).astype('float64')
salary = df.Salary

alphas = 10**np.linspace(5,-2,100)*0.5
coef = []

for x in alphas:
    lasso = Lasso(alpha=x,max_iter = 10000)
    lasso.fit(df_num,salary)
    coef.append(lasso.coef_)

ax = plt.gca()
ax.plot(alphas, coef)
ax.set_xscale('log')
plt.title('Lasso Regression on Salary')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()

df_coef = pd.DataFrame(coef, columns=df_num.columns,index=alphas)
print(df_coef)
### From the plot and coefficients dataframe we can see that the last three remaining predictors are:
### AtBat, CAtBat, CRBI



lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(df_num, salary)

lasso = Lasso()
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(df_num, salary)
print('------------------------------------------')
print('The optimal regularization penality:',lassocv.alpha_)
print(pd.Series(lasso.coef_, index=df_num.columns))
print('There are about 8 predictors left in the model')


##############Problem 1.2
alphas = 10**np.linspace(10,-2,100)*0.5
coef2 = []
for x in alphas:
    ridge = Ridge(alpha=x,max_iter = 10000)
    ridge.fit(df_num,salary)
    coef2.append(ridge.coef_)


bx = plt.gca()
bx.plot(alphas, coef2)
bx.set_xscale('log')
plt.title('Ridge Regression on Salary')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.show()


ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(df_num, salary)

ridge = Ridge()
ridge.set_params(alpha=ridgecv.alpha_)
ridge.fit(df_num, salary)
print('------------------------------------------')
print('The optimal regularization penality:',ridgecv.alpha_)
print(pd.Series(ridge.coef_, index=df_num.columns))
print('There are about 8 predictors left, similar to lasso')


##############Problem 2
### Vairiance-bias tradeoff refers to a property of the model:
### one can decrease variance by increasing bias, and vice versa

### Regularization provides a way to find the balancing point between variance and bias

### Models that overfits the data have low bias and high variance
### Models that underfits the data have high bias and low variance