import numpy as np
import pandas as pd
import pretty_midi
import os
#=======================================================================#以下自写
def get_genres(path):                                                   #get流派种类
    track_id = []
    genre = []
    with open(path) as f:
        line = f.readline()
        while line:
            if line[0] != '#':
                [x, y, *_] = line.strip().split("\t")
                track_id.append(x)
                genre.append(y)
            line = f.readline()
    genre_df = pd.DataFrame(data={"Genre": genre, "TrackID": track_id})
    return genre_df

def normalize(features):                                                #特征归一化
    tempo = (features[0] - 150) / 300
    num_sig_changes = (features[1] - 2) / 10
    resolution = (features[1] - 260) / 400
    time_sig_1 = (features[3] - 3) / 8
    time_sig_2 = (features[4] - 3) / 8
    #avg_tone_height = features[2] / 740
    #avg_tone_str = features[3] / 127
    return [tempo, num_sig_changes,resolution, time_sig_1, time_sig_2]
    #return [tempo,resolution,avg_tone_height,avg_tone_str]


def feather_process(features):                                          #异常值处理
    for i in len(features[0]):
        if(features[0]<=0 or features[0]>1 or
                features[1]<=0 or features[1]>1 or
                features[2]<=0 or features[2]>1 or
                features[3]<=0 or features[3]>1):
            del features[:][i]


def get_features(path):                                                 #Pretty_midi提取特征值
    midi_data = pretty_midi.PrettyMIDI(path)

    tempo = midi_data.estimate_tempo()                              #tempo

    num_sig_changes = len(midi_data.time_signature_changes)        #拍子变化次数
    resolution = midi_data.resolution                               #解析度
    ts_changes = midi_data.time_signature_changes
    #chroma = midi_data.get_chroma(1)                                #色度卷
    #piano_roll = midi_data.get_piano_roll(1)                        #钢琴卷
    #for i in chroma:
    #    for j in i:
    #        if(j!=0):
    #            tone_height = tone_height+j
    #            tone_height_num = tone_height_num+1
    #avg_tone_str = tone_height/tone_height_num                      #平均音强

    #fertz=[16.352,17.324,18.354,19.445,20.602,21.827,23.125,24.5,25.957,27.5,29.135,30.868]

    #for i in range[0:12]:
    #    for j in range[0:10]:
    #        for m in piano_roll[i*10+j]:
    #            if(m!=0):
    #                tone_str_num = tone_str_num+1
    #                tone_str = tone_str+fertz[j]*(2**i)

    #avg_tone_height = tone_str/tone_str_num                         #平均音高（赫兹表示）
    ts_1 = 4
    ts_2 = 4
    if len(ts_changes) > 0:
        ts_1 = ts_changes[0].numerator
        ts_2 = ts_changes[0].denominator
    return normalize([tempo, num_sig_changes, resolution, ts_1, ts_2]) #旋律，拍子变化次数，解析度，起始拍子
    #return normalize([tempo, resolution,avg_tone_height,avg_tone_str])  #旋律，解析度，平均音高，平均音强

#=======================================================================#以下参考内容
def get_matched_midi(midi_folder, genre_df):                            #midi匹配
    track_ids, file_paths = [], []
    for dir_name, subdir_list, file_list in os.walk(midi_folder):
        if len(dir_name) == 36:
            track_id = dir_name[18:]
            file_path_list = ["/".join([dir_name, file]) for file in file_list]
            for file_path in file_path_list:
                track_ids.append(track_id)
                file_paths.append(file_path)
    all_midi_df = pd.DataFrame({"TrackID": track_ids, "Path": file_paths})

    df = pd.merge(all_midi_df, genre_df, on='TrackID', how='inner')
    return df.drop(["TrackID"], axis=1)

def extract_midi_features(path_df):                                     #构造矩阵
    all_features = []
    for index, row in path_df.iterrows():
        features = get_features(row.Path)
        genre = label_dict[row.Genre]
        if features is not None:
            features.append(genre)
            all_features.append(features)
    return np.array(all_features)


def one_hot(labels):                                                    #One—Hot编码
    return np.eye(num_classes)[labels].astype(int)
#=======================================================================#以下自写
genre_path = "msd_tagtraum_cd1.cls"
#来源http://www.tagtraum.com/msd_genre_datasets.html
genre_df = get_genres(genre_path)                                       #构建流派df

label_list = list(set(genre_df.Genre))
label_dict = {lbl: label_list.index(lbl) for lbl in label_list}         #创建流派列表和字典
#输出测试
print(genre_df.head(), end="\n\n")
print(label_list, end="\n\n")
print(label_dict, end="\n\n")


midi_path = "lmd_matched"
#来源https://colinraffel.com/projects/lmd/
matched_midi_df = get_matched_midi(midi_path, genre_df)                 #midi匹配


labeled_features = extract_midi_features(matched_midi_df)               #构造矩阵
#输出测试
print(labeled_features)
print(matched_midi_df.head())

#=======================================================================#以下半参考半自写

labeled_features = np.random.permutation(labeled_features)              #随机分配

num = len(labeled_features)                                             #分训练集测试集验证集
num_training = int(num * 0.6)
num_validation = int(num * 0.8)
training_data = labeled_features[:num_training]
validation_data = labeled_features[num_training:num_validation]
test_data = labeled_features[num_validation:]

num_cols = training_data.shape[1] - 1                                   #按标签分离特征值
training_features = training_data[:, :num_cols]
validation_features = validation_data[:, :num_cols]
test_features = test_data[:, :num_cols]

num_classes = len(label_list)
training_labels = training_data[:, num_cols].astype(int)
validation_labels = validation_data[:, num_cols].astype(int)
test_labels = test_data[:, num_cols].astype(int)

#输出测试
print(test_features[:10])
print(test_labels[:10])
print(one_hot(test_labels)[:10])

