#数据预处理
import pandas as pd
from bs4 import BeautifulSoup

with open("/data/unlabeledTrainData.tsv", "r") as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

with open("/data/labeledTrainData.tsv", "r") as f:
    labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]

unlabel = pd.DataFrame(unlabeledTrain[1:], columns=unlabeledTrain[0])
label = pd.DataFrame(labeledTrain[1:], columns=labeledTrain[0])


def cleanReview(subject):
    # 数据处理函数
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',',
                                                                                                          '').replace(
        '.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)

    return newSubject

    unlabel["review"] = unlabel["review"].apply(cleanReview)
    label["review"] = label["review"].apply(cleanReview)

    # 将有标签的数据和无标签的数据合并
    newDf = pd.concat([unlabel["review"], label["review"]], axis=0)
    # 保存成txt文件
    newDf.to_csv("/data/preProcess/wordEmbdiing.txt", index=False)