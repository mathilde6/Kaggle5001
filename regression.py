import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MultiLabelBinarizer

def onehot_transform(x,transform_feature=None,save_name=None,**kwargs):
    classes = []
    if transform_feature == None:
        transform_feature = list(x.columns)
    for i in transform_feature:
        classes.extend(list(x[i].unique())) #classes是需要转换特征的所有类别取值
    clf = MultiLabelBinarizer(classes=classes, **kwargs)
    transform_data = clf.fit_transform(x[transform_feature].values)
    drop_data = x.drop(transform_feature, axis=1)
    classes = [str(cla) for cla in classes]
    heads = []
    for fea in transform_feature:  #遍历每一个特征
        n = x[fea].unique().shape[0]  #n为当前遍历的特征类别的数量
        head = [fea+'_'+item for item in classes[:n]]  #新编码的特征名称，由原特征与后缀拼接而成
        classes = classes[n:]
        heads.extend(head)

    con_x=pd.concat([drop_data,pd.DataFrame(transform_data, columns = heads)], axis=1)
    if save_name:
        joblib.dump(clf, save_name + '.pkl')
    return con_x
	
y_train = pd.read_csv('train.csv')
y_train = pd.DataFrame(y_train.iloc[:,-1])

X_train = pd.read_csv('train.csv')
X_train = X_train.iloc[:,1:-1]

X_test = pd.read_csv('test.csv')
X_test = X_test.iloc[:,1:]

X = pd.concat([X_train,X_test])
X = X.reset_index(drop=True)

X = onehot_transform(X,transform_feature=['penalty'])

X_train = X.iloc[:400,:]
X_test = X.iloc[400:,:]
X_test = X_test.reset_index(drop=True)

clf = DecisionTreeRegressor()
clf.fit(X_train,y_train)

preds = clf.predict(X_test)

preds_df = pd.DataFrame([list(preds)]).T
preds_df.columns = ['time']
preds_df.to_csv('predict_result9.csv')