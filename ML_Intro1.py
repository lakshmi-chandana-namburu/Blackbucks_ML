import pandas as pd
import pydotplus
os.environ["PATH"] += os.pathsep + 'C:/Users/Lenovo/Downloads/Graphviz/bin/'
tt1 = pd.get_dummies(tt,columns=['Pclass', 'Sex', 'Embarked'])
tt1.shape
tt1.info()
tt1.describe()
X_train = tt1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'], 1)
X_train.info()
y_train = tt['Survived']
X_train.info()
dt = tree.DecisionTreeClassifier(criterion = 'entropy')
dt.fit(X_train,y_train)

dot_data = io.StringIO()  
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
os.getcwd()
graph.write_pdf("DT.pdf")

ttest=pd.read_csv(r"D:\20_594\blackbucks_level2\titanic_dataset\test.csv")
ttest.shape
ttest.info()
ttest.Fare[ttest['Fare'].isnull()] = ttest['Fare'].mean()
titanic_test1 = pd.get_dummies(ttest, columns=['Pclass', 'Sex', 'Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
ttest['Survived'] = dt.predict(X_test)
os.getcwd()
ttest.to_csv(r"D:\20_594\blackbucks_level2\Submission_Attempt2.csv", columns=['PassengerId', 'Survived'], index=False)
