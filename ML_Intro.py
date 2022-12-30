import os
import io
import pydotplus
import pandas as pd
from sklearn import tree
os.environ["PATH"] += os.pathsep + 'C:/Users/Lenovo/Downloads/Graphviz/bin/'
tt=pd.read_csv(r"D:\20_594\blackbucks_level2\titanic_dataset\train.csv")
xtt=tt[['Pclass','SibSp','Parch']]
ytt=tt['Survived']
dt=tree.DecisionTreeClassifier()
dt.fit(xtt,ytt)
ob=io.StringIO()
tree.export_graphviz(dt,out_file=ob,feature_names=xtt.columns)
file1=pydotplus.graph_from_dot_data(ob.getvalue())
os.getcwd()
file1.write_pdf("decisiontree1.pdf")
ttest=pd.read_csv(r"D:\20_594\blackbucks_level2\titanic_dataset\test.csv")
xtest=ttest[['Pclass','SibSp','Parch']]
ttest['Survived']=dt.predict(xtest)
ttest.to_csv(r"D:\20_594\blackbucks_level2\submission_tit.csv",columns=['PassengerId','Survived'],index=False)
