from preprogress import  get_features,label_list
from sklearn.externals import joblib
#=======================================================================#以下自写
def make_prediction(clf, midi_path):                                    #对单个midi进行预测
    features = get_features(midi_path)
    prediction_ind = list(clf.predict([features])[0]).index(1)
    acc=clf.predict_proba([features])
    #prediction = label_list[prediction_ind]
    print(label_list[prediction_ind])
    print(acc)


#单个预测
test_midi_path = "lmd_matched/B/F/E/TRBFELB128F426BFF2/289270d85c81802d912c9907c645dc2d.mid"
#print(make_prediction(classifier, test_midi_path))
clf = joblib.load('save/clf.pkl')                                        #读取model
make_prediction(clf, test_midi_path)