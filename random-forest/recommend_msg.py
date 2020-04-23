from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus

import numpy as np 
import pandas as pd 

if __name__ == "__main__":
    with open("./论文数据.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    # print(lines)
    list_lines = []
    for line in lines :
        list_lines.append(line.rstrip("\\\n").split("\t"))
    # print(list_lines)

    src_train = []
    tgt_train = []
    for line in list_lines:
        src_train.append(line[:3])
        tgt_train.append(line[3])
    print(src_train)
    print(tgt_train)
    
    src_train = np.array(src_train)
    tgt_train = np.array(tgt_train)
    # X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], test_size=0.1)

    rf = RandomForestClassifier(n_estimators=3, random_state=0)
    rf.fit(src_train, tgt_train)

    print(rf.predict(np.array([[3, 3, 60]])))
    # print(rf.score(X_test, y_test))

    clf = rf.estimators_[1]## 取出一颗树
    # dot_data = tree.export_graphviz(clf, out_file=None) 
    dot_tree = tree.export_graphviz(clf,out_file=None,feature_names=["提交错误次数", "平均评测点错误个数", "答题时间"],class_names=["错误评测点", "解题思路", "参考答案", "不做任何推荐"],filled=True, rounded=True,special_characters=True)

    graph = pydotplus.graph_from_dot_data(dot_tree) 
    graph.write_png("tree1.png")




    