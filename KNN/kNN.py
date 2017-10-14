# -*- coding: UTF-8 -*-  

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5

    sortedDistIndicies = distances.argsort()
    classCount = {}  
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), 
                              reverse=True)

    return sortedClassCount[0][0]


# 将文本记录到转换NumPy的解析程序

def file2matrix(filename):

    # 得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()  # 读取整个文件到一个迭代器以供我们遍历
    numberOfLines = len(arrayOLines)

    # 创建返回的NumPy矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    # 解析文件数据到列表
    for line in arrayOLines:
        line = line.strip()  # 用于移除字符串头尾指定的字符（默认为空格）                              #  截取掉所有的回车字符
        listFromLine = line.split('\t') # 通过指定分隔符对字符串进行切片，                                        # 如果参数num 有指定值，
                                        # 则仅分隔 num 个子字符串
                                        # 使用tab字符\t将上一步得到的整行                                         # 数据分割成一个元素列表
        returnMat[index, :] = listFromLine[0: 3]  # 选取前三个元素存储到特                                                  # 征矩阵
        classLabelVector.append(int(listFromLine[-1])) # 最后一列
        index += 1

    return returnMat, classLabelVector


# 归一化特征值

def autoNorm(dataSet):
    minVals = dataSet.min(0)  # axis=0 每列的最小值
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 第一维度的长度
    normDataSet = dataSet - tile(minVals, (m, 1))

    # 特征值相除
    normDataSet = normDataSet/tile(ranges, (m, 1))

    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10  # 数据集中用来做测试数据的比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],\
                normMat[numTestVecs:m, :],\
                datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: {}, the real answer is: {}"
                .format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: {}"\
            .format(errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input(\
            "percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])

    classifierResult = classify0((inArr-minVals)/ranges, normMat,\
            datingLabels, 3)
    print("You will probably like this person: ",\
            resultList[classifierResult - 1])
