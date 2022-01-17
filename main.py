# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:39:29 2021

@author: mpica
"""
# 0. import packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf\
    
# models
import sklearn as sk
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LassoCV,Lasso, LinearRegression, LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif, chi2
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

%matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#%%
# 1. Read data
df_train = pd.read_csv('C:/Users/mpica/Pictures/Documentos/kaggle/titanic/train.csv')
df_test = pd.read_csv('C:/Users/mpica/Pictures/Documentos/kaggle/titanic/test.csv')

#%%
# 2. Explore data
print(df_train.dtypes)
print(df_train.head())
print(df_train.isna().sum())
print(df_train.describe())
print(df_train.Embarked.value_counts())

target_col = 'Survived'

#%%
# 3. Visulaize data
sns.distplot(df_train[df_train.Survived==1].Fare, bins = 20, color='green', kde=False)
sns.distplot(df_train[df_train.Survived==0].Fare, bins = 20, color='orange', kde=False)

#%%
# 4. Data cleaning and new features

# impute NA in categorical
col_objects = df_train.columns[df_train.dtypes == 'object']
col_num = df_train.columns[df_train.dtypes != 'object']
for col in col_objects:
    df_train.loc[df_train[col].isna(),col] = 'None'
    df_test.loc[df_test[col].isna(),col] = 'None'

# new numerical from categorical
df_train['new_male'] = df_train['Sex'].map(lambda x: x in ['male']).astype('int64')
df_train['new_floor'] = df_train['Cabin'].str[0] # first letter of cabin, assumed to be floor
df_train['new_hasCabin'] = (df_train['Cabin'] == 'None').astype('int64')
# df_train['new_title'] = df_train.Name.map(lambda x: x.split(',')[1].split(' ')[1])
df_train['new_title'] = df_train.Name.map(lambda x: x.split(',')[1].split('.')[0][1:])
df_train['new_crew'] = df_train.new_title.map(lambda x: x in ['Capt','Col','Major','Rev']).astype('int64')
df_train['new_royalty'] = df_train.new_title.map(lambda x: x in ['Jonkheer','Lady','Sir','the Countess','Don','Dona']).astype('int64') 
df_train['new_noAge'] = (df_train['Age'].isna()).astype('int64')
# df_train['Fare'] = np.log(df_train['Fare']+0.1)

# impute NA in numerical, only Age
auxAge = df_train.groupby('new_title').apply(lambda df: pd.Series({
            'count':len(df['Age']),
            'meanAge':np.mean(df['Age']),
            'medianAge':np.median(df['Age']),
            'isnaAge':df['Age'].isna().sum()
            }))
auxFare = df_train.groupby('Pclass').apply(lambda df: pd.Series({
            'count':len(df['Fare']),
            'meanFare':np.mean(df['Fare']),
            'medianFare':np.median(df['Fare']),
            'isnaFare':df['Fare'].isna().sum()
            }))


df_train.loc[df_train.Age.isna(),'Age'] = df_train.new_title.map(lambda x: auxAge.meanAge.values[auxAge.index==x][0])[df_train.Age.isna()].astype('float64') 

# for col in list(set(col_num)-set([target_col])):
#     print(col)
#     df_train.loc[df_train[col].isna(),col] = df_train[col].median()
#     df_test.loc[df_test[col].isna(),col] = df_train[col].median() # use meadian of train, no leckeage
    
# df_train = df_train.dropna()

OneHotEncode = OneHotEncoder(sparse=False)
col_num = list(set(df_train.columns[df_train.dtypes != 'object']) - set(['PassengerId']))
col_num = [i for i in col_num if i != target_col]
ordEncode_col = ['Embarked']
ordEncoded_train = pd.DataFrame(OneHotEncode.fit_transform(df_train[ordEncode_col]))
ordEncoded_train.columns = [ordEncode_col[0]+str(i)[2:] for i in OneHotEncode.get_feature_names()]
ordEncoded_train.index = df_train.index

X_train = pd.concat([df_train[col_num], ordEncoded_train],axis=1)
y_train = df_train[target_col].astype(int)

#%%
# 5. try and validate multiple models and parameters


# model: random forest, test number of trees and minimum number of elements in leafs
scores_all = []
for n_estimators in [10,50,100]:
    scores = cross_val_score(RandomForestClassifier(n_estimators = n_estimators, min_samples_leaf=10, random_state=0),
                            X_train,y_train,cv=10,scoring='accuracy') # scoring = 'roc_auc'
    scores_all.append(np.mean(scores))
print(scores_all)
    
scores_all = []
for min_leaf in [1,2,5]:
    scores = cross_val_score(RandomForestClassifier(n_estimators = 100, min_samples_leaf=min_leaf, random_state=0),
                            X_train,y_train,cv=10,scoring='accuracy')
    scores_all.append(np.mean(scores))
print(scores_all) 

GridSearchRF=GridSearchCV(estimator=RandomForestClassifier(random_state=0), 
                          param_grid={'n_estimators':[50,100,200],
                                      'min_samples_leaf':[2,5],
                                      'criterion':['gini','entropy']}, 
                          scoring='accuracy',cv=3)
score=cross_val_score(GridSearchRF,X_train,y_train,scoring='accuracy',cv=10)

# model: gradient boosting, test number of trees and minimum number of elements in leafs
scores_all = []
for n_estimators in [50,100,150,200]:
    scores = cross_val_score(GradientBoostingClassifier(n_estimators = n_estimators, random_state=0),
                            X_train,y_train,cv=10,scoring='accuracy')
    scores_all.append(np.mean(scores))
print(scores_all) 

scores_all = []
for learning_rate in [0.05,0.1,0.3,0.5]:
    scores = cross_val_score(GradientBoostingClassifier(n_estimators = 200, random_state=0, learning_rate = learning_rate),
                            X_train,y_train,cv=10,scoring='accuracy')
    scores_all.append(np.mean(scores))
print(scores_all) 

cross_val_score(GradientBoostingClassifier(n_estimators = 200, random_state=0, learning_rate = 0.1),
                            X_train,y_train,cv=10,scoring='accuracy').mean()

# model support vector machines, test violation penalization and rbf kernel gamma
svm = make_pipeline(StandardScaler(),SVC(random_state=0))
r=[0.0001,0.001,0.1,1,10,50,100]
PSVM=[{'svc__C':r, 'svc__kernel':['linear']},
      {'svc__C':r, 'svc__gamma':r, 'svc__kernel':['rbf']}]
GSSVM=GridSearchCV(estimator = svm, 
                   param_grid=PSVM, scoring='accuracy', cv=2)
score = cross_val_score(GSSVM, X_train.astype(float), y_train,scoring='accuracy', cv=5).mean()

# model: lasso, test penalization parameter alpha
scores_all = []
for alpha in np.logspace(-5, 1, 7):
    scores = cross_val_score(Lasso(random_state=0,alpha=alpha),X_train,y_train,cv=10,scoring='roc_auc')
    scores_all.append(np.mean(scores))
print(scores_all) 

scores = cross_val_score(LassoCV(),X_train,y_train,cv=10,scoring='roc_auc')
print(np.mean(scores)) 
model = LassoCV(random_state=0, cv=10)
model.fit(X_train,y_train)

# test several models together
models = {'randomforest': RandomForestClassifier(n_estimators = 100, min_samples_leaf=2, random_state=0),
          'boosting tree': GradientBoostingClassifier(n_estimators = 100, loss = 'exponential', random_state=0),
          'lasso': LassoCV(cv=10,random_state=0),
          'logistic': LogisticRegression(random_state=0)}
scores_all = []
for model_name in models:
    scores = -1*cross_val_score(models[model_name],X_train,y_train,cv=10,scoring='roc_auc')
    scores_all.append(np.mean(scores))
print(scores_all)

# model: xgboost
xgb.cv(nfold = 10, 
       dtrain = xgb.DMatrix(X_train,label=y_train),
       metrics = ['auc','logloss'],
       params = {'booster': 'gbtree',
                 'learning_rate': 0.3,
                 'max_depth':3,
                 'min_child_weight':10},
       num_boost_round = 100,
       early_stopping_rounds=50).mean()

# n_train = int(X_train.shape[0]*0.95)
# cv_scores = []
# for i in range(10):
#     shuffle = np.random.choice(X_train.shape[0],X_train.shape[0], replace=False)
#     idx_train, idx_test = shuffle[:n_train], shuffle[n_train:]
#     model = xgb.XGBClassifier(objective = 'binary:logistic')
#     model.fit(X_train.loc[idx_train,:],y_train[idx_train])


#%%
# 6. Submission
df_test['new_male'] = df_test['Sex'].map(lambda x: x in ['male']).astype('int64')
df_test['new_floor'] = df_test['Cabin'].str[0] # first letter of cabin, assumed to be floor
df_test['new_hasCabin'] = (df_test['Cabin'] == 'None').astype('int64')
df_test['new_title'] = df_test.Name.map(lambda x: x.split(',')[1].split('.')[0][1:])
df_test['new_crew'] = df_test.new_title.map(lambda x: x in ['Capt','Col','Major','Rev']).astype('int64')
df_test['new_royalty'] = df_test.new_title.map(lambda x: x in ['Jonkheer','Lady','Sir','the Countess','Don','Dona']).astype('int64') 
df_test['new_noAge'] = (df_test['Age'].isna()).astype('int64')
df_test.loc[df_test.Age.isna(),'Age'] = df_test.new_title[df_test.Age.isna()].map(lambda x: auxAge.meanAge.values[auxAge.index==x][0]).astype('float64')
df_test.loc[df_test.Fare.isna(),'Fare'] = df_test.Pclass[df_test.Fare.isna()].map(lambda x: auxFare.medianFare.values[auxFare.index==x][0]).astype('float64')

ordEncoded_test = pd.DataFrame(OneHotEncode.transform(df_test[ordEncode_col]))
ordEncoded_test.columns = [ordEncode_col[0]+str(i)[2:] for i in OneHotEncode.get_feature_names()]
ordEncoded_test.index = df_test.index

X_test = pd.concat([df_test[col_num], ordEncoded_test],axis=1)

model = GradientBoostingClassifier(n_estimators = 50, random_state=0, learning_rate = 0.1)
model.fit(X_train,y_train)

pd.DataFrame({'PassengerId': df_test['PassengerId'].astype('int'), 'Survived':model.predict(X_test)}).to_csv('C:/Users/mpica/Pictures/Documentos/kaggle/titanic/submission.csv', index = False)

#%%
# 7. test data analysis & check stuff

# 7.1 data aggregation
# check ordinal encoder
aux = pd.concat([df_train, ordEncoded_train],axis=1)
res = aux.groupby(['Embarked']).apply(lambda df: pd.Series({
    'Cmin':min(df['Embarked_C']),
    'Cmax':max(df['Embarked_C']),
    'Nonemin':min(df['Embarked_None']),
    'Nonemax':max(df['Embarked_None']),
    'Qmin':min(df['Embarked_Q']),
    'Qmax':max(df['Embarked_Q']),
    'Smin':min(df['Embarked_S']),
    'Smax':max(df['Embarked_S'])
    }))
aux.groupby(['Embarked']).Embarked_C.agg([min,len,'mean'])
aux.groupby(['Embarked']).agg({'Embarked_C':[min,len,'mean'],'Embarked_None':[min,len,'mean']})

aux.groupby(['Pclass']).apply(lambda df: pd.Series({
    'Fare_mean':np.mean(df['Fare']),
    'Fare_std':np.std(df['Fare'])
    }))

aux.groupby(['Pclass','Embarked']).agg({
    'PassengerId':['count'],
    'Fare':['mean','median','std'],
    'Survived':['mean','std',]})

aux.groupby(['Survived']).Pclass.agg(['value_counts'])
aux.groupby(['Survived','Pclass']).size()

# 7.2 statistical tests
# 7.2.1 two discrete variables: pearson chi2
res = aux.groupby(['Survived']).apply(lambda df: pd.Series({
    'Pclass1':sum(df['Pclass']==1),
    'Pclass2':sum(df['Pclass']==2),
    'Pclass3':sum(df['Pclass']==3),
    'total':df['Pclass'].count()
    }))
# better with crosstab
obs = np.array(pd.crosstab(aux['Survived'],aux['Pclass']))
sts,p,dof,exp = stats.chi2_contingency(pd.crosstab(aux['Survived'],aux['Pclass']))
stats.chisquare(obs.flatten(),exp.flatten(),dof) # test
1-stats.chi2.cdf(sts,dof) # test
print(sts == np.sum((obs.flatten()-exp.flatten())**2/exp.flatten()))

# 7.2.2 continious-discrete variables: Wald same mean
aux.groupby(['Survived']).Fare.agg(['mean','std','count'])
se = np.sqrt(31.39**2/549+66.60**2/342)
sts = (48.4-22.12)/se
(1-stats.norm.cdf(sts))*2

aux.groupby(['Survived']).Parch.agg(['mean','std','count'])
se = np.sqrt(0.823**2/549+0.771**2/342)
sts = np.abs((0.330-0.465)/se)
(1-stats.norm.cdf(sts))*2

aux.groupby(['Survived']).Age.agg(['mean','std','count'])
se = np.sqrt(12.5**2/549+13.8**2/342)
sts = np.abs((30.03-28.29)/se)
(1-stats.norm.cdf(sts))*2

# better with the p-value of coefficient in univariate regression statistics (see also next)
f,p = f_classif(X_train,y_train)
chi,p = chi2(X_train,y_train)

# 7.3 statistical inference on linear models
X_train_pIndep = sm.add_constant(X_train)
model = sm.GLM(y_train, X_train_pIndep, family=sm.families.Binomial()).fit()
model.summary()
model2 = sm.Logit(y_train, X_train_pIndep).fit()

Xy_train = X_train.copy()
Xy_train['Survived'] = y_train
model1p = smf.glm(formula = 'Survived ~  Pclass + Age + SibSp + Parch + Fare + C(new_male) + Embarked_C+Embarked_None+Embarked_Q', 
                 data = Xy_train, family=sm.families.Binomial()).fit()
model2p = smf.logit(formula = 'Survived ~  Pclass + Age + SibSp + Parch + Fare + C(new_male) + Embarked_C+Embarked_None+Embarked_Q', 
                 data = Xy_train).fit()
model1pp = smf.glm(formula = 'Survived ~  Pclass + Age + SibSp + Parch + Fare + C(new_male) + C(Embarked)', 
                 data = df_train, family=sm.families.Binomial()).fit()

# 7.4 random generation
# random integer, for bootstrapping
stats.randint.rvs(0,1000,size=10)
np.random.choice(1000,size=10) #replace = True
np.random.choice(1000,size=10,replace=False)
# random uniform -1 1
stats.uniform.rvs(-1,1,size=10)
np.random.uniform(-1,1,10)
# random normal

# 7.5 cluster features
kmeans = KMeans(n_clusters=5)
kmeans.fit(X_train[['Pclass','Fare','Age','Embarked0']])