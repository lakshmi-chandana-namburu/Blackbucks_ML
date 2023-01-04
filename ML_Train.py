import pandas as pd
import os
from sklearn.impute import SimpleImputer # used to fill missing data
from sklearn import tree
from sklearn import model_selection
tt=pd.read_csv(r"D:\20_594\blackbucks_level2\titanic_dataset\train.csv")
ttest=pd.read_csv(r"D:\20_594\blackbucks_level2\titanic_dataset\test.csv")
ttest.Survived=None
t=pd.concat([tt,ttest])
t.shape
t.info()
def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()
t['Title']=t['Name'].map(extract_title)
mean_imputer=SimpleImputer()
mean_imputer.fit(tt[['Age','Fare']])
t[['Age','Fare']]=mean_imputer.transform(t[['Age','Fare']])
def convert_age(age):
    if(age>=0 and age<=10):
        return 'Child'
    elif(age <= 25): 
        return 'Young'
    elif(age <= 50): 
        return 'Middle'
    else: 
        return 'Old'
#Convert numerical Age column to categorical Age_Cat column
t['Age_Cat'] = t['Age'].map(convert_age)
t['FamilySize'] = t['SibSp'] +  t['Parch'] + 1
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
t['FamilySize_Cat'] = t['FamilySize'].map(convert_familysize)
titanic1 = pd.get_dummies(t, columns=['Sex','Pclass','Embarked', 'Age_Cat', 'Title', 'FamilySize_Cat'])
titanic2 = titanic1.drop(['PassengerId','Name','Age','Ticket','Cabin','Survived'], axis=1, inplace=False)
X_train = titanic2[0:891] #0 t0 891 records
X_train.shape
X_train.info()
y_train = tt['Survived']
tree_estimator = tree.DecisionTreeClassifier()
dt_grid = {'max_depth':[15, 16, 17], 'min_samples_split':[2,3], 'criterion':['gini','entropy']}
param_grid = model_selection.GridSearchCV(tree_estimator, dt_grid, cv=15) #Evolution of tree cv - into how many records/groups we have to divide when there is no test data
param_grid.fit(X_train, y_train) #Building the tree
print(param_grid.cv_results_)
print(param_grid.best_score_) #Best score
print(param_grid.best_params_)
fi_df = pd.DataFrame({'feature':X_train.columns, 'importance':  param_grid.best_estimator_.feature_importances_}) #You may notice that feature	importance "Title_Mr" has more importance
print(fi_df)
X_test = titanic2[tt.shape[0]:] 
# X_test.shape
# X_test.info()
ttest['Survived'] = param_grid.predict(X_test)
ttest.to_csv(r'D:\20_594\blackbucks_level2\titanic_dataset\Attempt_Params_CV.csv', columns=['PassengerId','Survived'],index=False)
