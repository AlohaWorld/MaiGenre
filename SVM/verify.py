import numpy as np
from preprogress import one_hot,test_features,test_labels
from sklearn.svm import SVC
from model import classifier
#=======================================================================#以下参考
def calculate_accuracy(clf, t_features, t_labels):      #计算Acc
    count = 0
    predictions = clf.predict(t_features)
    t_labels_hot = one_hot(t_labels)
    for i in range(len(t_features)):
        if (type(clf) == SVC):
            if t_labels[i] == predictions[i]:
                count += 1
        else:
            if np.array_equal(t_labels_hot[i], predictions[i]):
                count += 1
    return count / len(t_features)


# 由测试集得到的Acc
print(calculate_accuracy(classifier, test_features, test_labels))