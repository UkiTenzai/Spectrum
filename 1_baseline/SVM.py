import numpy as np
from scipy.io import loadmat
import pandas as pd
import random
from sklearn.metrics import accuracy_score
import joblib
from sklearn import svm

# 读取高光谱数据和标签（gt）
input_image = loadmat("Dataset_Image_University.mat")
gt = loadmat("University_groundtruth_map.mat")
input_image = input_image["DataTest"]
gt = gt["map"]
input_image = input_image.reshape(610, 340, 103)
print(input_image.shape)


# 构造数据集
dict_k = {}
traindata = []
trainlabel = []
classtrain = [[], [], [], [], [], [], [], [], []]
for i in range(gt.shape[0]):
    for j in range(gt.shape[1]):
        if gt[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if gt[i][j] not in dict_k:
                dict_k[gt[i][j]] = 0
            dict_k[gt[i][j]] += 1
            trainlabel.append(gt[i][j].tolist())
            traindata.append(input_image[i][j].tolist())
            classtrain[gt[i][j] - 1].append(input_image[i][j].tolist())

        else:
            pass

# 输出数据集信息
print("各类别数量如下：{}".format(dict_k))
print("总数：", sum(dict_k.values()))
traindata = np.array(traindata)
trainlabel = np.array(trainlabel)
trainlabel = trainlabel - 1  # 忽略0类

# 打乱顺序
for i in range(9):
    random.shuffle(classtrain[i])

# 每一类选出一定数目个作为训练集
csvtrain = []
csvlabel = []
for i in range(9):
    for j in range(50):
        csvtrain.append(classtrain[i][j])
        csvlabel.append(i)
csvtrain = np.array(csvtrain)
csvlabel = np.array(csvlabel)

# 小部分数据作为训练集
csvlabel = np.reshape(csvlabel, (csvlabel.shape[0], 1))
csvtrain = np.concatenate((csvtrain, csvlabel), axis=1)
csvtrain = pd.DataFrame(csvtrain)

# 全部数据作为测试集
trainlabel = np.reshape(trainlabel, (trainlabel.shape[0], 1))
csvtest = np.concatenate((traindata, trainlabel), axis=1)
csvtest = pd.DataFrame(csvtest)

# 重命名列
p_col = []
for i in range(csvtrain.shape[1]):
    p_col.append("x{}".format(i))
p_col[-1] = "label"
csvtrain.columns = p_col
csvtest.columns = p_col

# 讲数据写到csv中 方便不同模型调用
csvtrain.to_csv("train.csv", index=False)
csvtest.to_csv("test.csv", index=False)

# 读取训练集
csvtrain = pd.read_csv("train.csv")
y_train = csvtrain["label"].values
X_train = csvtrain.drop(["label"], axis=1).values

# 读取测试集合
csvtest = pd.read_csv("test.csv")
y_test = csvtest["label"].values
x_test = csvtest.drop(["label"], axis=1).values

model = svm.SVC(kernel="linear", C=0.1, decision_function_shape="ovo")
model.fit(X_train, y_train)
joblib.dump(model, "SVM_model.m")
print("模型保存！")
train_result = model.predict(x_test)
accuracy = accuracy_score(y_test, train_result)
print("Accuracy: ", accuracy)
