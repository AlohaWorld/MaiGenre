from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np
from preprogress import one_hot,training_features,training_labels,validation_features,validation_labels
from sklearn.externals import joblib
#=======================================================================#以下自写
def train_model(t_features, t_labels, v_features, v_labels):            #利用scikit-learn训练模型
    #！！模型及参数（改进）
    clf_1 = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(5, 5), random_state=1)
    clf_2 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 10), random_state=1)
    clf_3 = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 100), random_state=1)
    clf_svm = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0 , tol=0.001, cache_size=200, max_iter=-1, random_state=None)

    best_clf = None
    best_accuracy = 0

    #得到Acc最好的模型
    for clf in [clf_1, clf_2, clf_3, clf_svm]:
        t_labels_hot = one_hot(t_labels)
        v_labels_hot = one_hot(v_labels)
        if (type(clf) == SVC):
            clf = clf.fit(t_features, t_labels)
        else:
            clf = clf.fit(t_features, t_labels_hot)
        predictions = clf.predict(v_features)
        count = 0
        for i in range(len(v_labels)):
            if (type(clf) != SVC):
                if np.array_equal(v_labels_hot[i], predictions[i]):
                    count += 1
            else:
                if v_labels[i] == predictions[i]:
                    count += 1
        accuracy = count / len(v_labels_hot)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clf = clf

    print("Best Accuracy:", best_accuracy)
    joblib.dump(best_clf, 'save/clf1.pkl')                              #保存model
    return best_clf

classifier = train_model(training_features, training_labels, validation_features, validation_labels)