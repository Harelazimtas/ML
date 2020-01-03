import numpy as np
import librosa
from scipy import stats
import os

class fileSound():
    def __init__(self, mffc, number):
        self.mffc = mffc
        self.number = number
        self.dist = 0


def get_data():
    nodes = []
    for dir in os.listdir("train_data"):
        if not ".DS_Store" in dir:
            for file in os.listdir("train_data/" + dir):
                if file.endswith('.wav'):
                    y, sr = librosa.load("train_data/" + dir + "/" + file, sr=None)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr)
                    mfcc = stats.zscore(mfcc, axis=1)
                    nodes.append(fileSound(mfcc, str(dir)))

    return nodes


def distance(mffc, nodes):
    for i in nodes:
        dist = np.linalg.norm(abs(i.mffc-mffc))
        i.dist = dist


def predict(data, k):
    count = [0, 0, 0, 0, 0]
    data.sort(key=lambda x: x.dist,reverse=False)
    for index in range(len(data)):
        if k == index:
            break
        if data[index].number == "one":
            count[0] += 1
        elif data[index].number == "two":
            count[1] += 1
        elif data[index].number == "three":
            count[2] += 1
        elif data[index].number == "four":
            count[3] += 1
        else:
            count[4] += 1
    number = count.index(max(count))+ 1
    return number


def knn(data,k):
    predict_file = open("output.txt", "w")
    for file in os.listdir("test_files"):
        if file.endswith('.wav'):
            y, sr = librosa.load("test_files/" + file, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc = stats.zscore(mfcc, axis=1)
            distance(mfcc, data)
            number_prediction = predict(data, k)
            predict_file.write(file+" - "+str(number_prediction)+"\n")
    predict_file.close()


def main():
    k = 1
    data = get_data()
    knn(data, k)


if __name__ == '__main__':
    main()
