
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 

from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus

import numpy as np 
import pandas as pd 


if __name__ == "__main__":
    iris = load_iris()
    # df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # print(df)
    # print(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], test_size=0.1)

    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rf.fit(X_train, y_train)

    # print(rf.predict(np.array([[5.9, 3.0, 5.1, 1.8]])))
    print(rf.score(X_test, y_test))

    # clf = rf.estimators_[5]## 取出一颗树
    # dot_data = tree.export_graphviz(clf, out_file=None) 
    # graph = pydotplus.graph_from_dot_data(dot_data) 
    # graph.write_png("tree.png")



    