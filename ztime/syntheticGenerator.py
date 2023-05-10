"""
Z-Time synthetic generator module
=====================
This synthetic generator is used to create our three synthetic datasets (SYN1, SYN2, and SYN3).
"""
import numpy as np

def generateSample(d1, d2):
    d1 = d1 + np.random.normal(0, 0.1, d1.shape)
    d2 = d2 + np.random.normal(0, 0.1, d1.shape)
    return np.array([d1, d2])

def generateClass(d1, d2, num=100, classno = 1):
    return np.array([generateSample(d1, d2) for i in range(num)]), [classno for i in range(num)]

def generateSampleTwoDim(length, noiseSize= 10):
    d1 = np.zeros(length)
    d2 = np.zeros(length)
    d3 = np.zeros(length)
    d4 = np.zeros(length)

    movement1 = np.random.randint(0, d1.shape[0])
    d1[movement1] = noiseSize

    movement2 = np.random.randint(movement1+1, d2.shape[0])
    d2[movement2] = -noiseSize

    d1 = d1 + np.random.normal(0, 0.1, length)
    d2 = d2 + np.random.normal(0, 0.1, length)

    return np.array([d1, d2])

def generateClassTwoDIm(length = 300, num=100, classno = 1, noiseSize=10):
    count = 0
    val = []
    while True:
        if count == num:
            return np.array(val), [classno for i in range(num)]
        else:
            try:
                val.append(generateSampleTwoDim(length, noiseSize))
                count += 1
            except:
                continue

def generateSampleTwoDim2(length, noiseSize= 10):
    d1 = np.zeros(length)
    d2 = np.zeros(length)

    movement1 = np.random.randint(0, d1.shape[0])
    d1[movement1] = -noiseSize

    movement2 = np.random.randint(movement1+1, d2.shape[0])
    d2[movement2] = noiseSize

    d1 = d1 + np.random.normal(0, 0.1, length)
    d2 = d2 + np.random.normal(0, 0.1, length)

    return np.array([d2,d1])

def generateClassTwoDIm2(length = 300, num=100, classno = 1, noiseSize=10):
    count = 0
    val = []
    while True:
        if count == num:
            return np.array(val), [classno for i in range(num)]
        else:
            try:
                val.append(generateSampleTwoDim2(length, noiseSize))
                count += 1
            except:
                continue

def generateSample2(length, noiseSize= 10):
    d1 = np.zeros(length)
    d2 = np.zeros(length)
    d3 = np.zeros(length)
    d4 = np.zeros(length)

    movement1 = np.random.randint(0, d1.shape[0])
    d1[movement1] = noiseSize

    movement2 = np.random.randint(movement1+1, d2.shape[0])
    d2[movement2] = -noiseSize

    movement3 = np.random.randint(movement2+1, d3.shape[0])
    d3[movement3] = noiseSize

    movement4 = np.random.randint(movement3+1, d3.shape[0])
    d4[movement4] = -noiseSize

    d1 = d1 + np.random.normal(0, 0.1, length)
    d2 = d2 + np.random.normal(0, 0.1, length)
    d3 = d3 + np.random.normal(0, 0.1, length)
    d4 = d4 + np.random.normal(0, 0.1, length)

    return np.array([d1, d2, d3, d4])

def generateClass2(length = 300, num=100, classno = 1, noiseSize=10):
    count = 0
    val = []
    while True:
        if count == num:
            return np.array(val), [classno for i in range(num)]
        else:
            try:
                val.append(generateSample2(length, noiseSize))
                count += 1
            except:
                continue

def generateSample3(length, noiseSize=10):
    d1 = np.zeros(length)
    d2 = np.zeros(length)
    d3 = np.zeros(length)
    d4 = np.zeros(length)

    movement1 = np.random.randint(0, d1.shape[0])
    d1[movement1] = noiseSize

    movement2 = np.random.randint(movement1+1, d2.shape[0])
    d2[movement2] = -noiseSize

    movement3 = np.random.randint(movement2+1, d3.shape[0])
    d3[movement3] = noiseSize

    movement4 = np.random.randint(movement3+1, d3.shape[0])
    d4[movement4] = -noiseSize

    d1 = d1 + np.random.normal(0, 0.1, length)
    d2 = d2 + np.random.normal(0, 0.1, length)
    d3 = d3 + np.random.normal(0, 0.1, length)
    d4 = d4 + np.random.normal(0, 0.1, length)
    return np.array([d3, d2, d1, d4])

def generateClass3(length = 300, num=100, classno = 1, noiseSize=10):
    count = 0
    val = []
    while True:
        if count == num:
            return np.array(val), [classno for i in range(num)]
        else:
            try:
                val.append(generateSample3(length, noiseSize))
                count += 1
            except:
                continue

def generateSample4(length, noiseSize=10):
    d1 = np.zeros(length)
    d2 = np.zeros(length)
    d3 = np.zeros(length)
    d4 = np.zeros(length)

    movement1 = np.random.randint(0, d1.shape[0])
    d1[movement1] = noiseSize

    movement2 = np.random.randint(movement1+1, d2.shape[0])
    d2[movement2] = -noiseSize

    movement3 = np.random.randint(movement2+1, d3.shape[0])
    d3[movement3] = noiseSize

    movement4 = np.random.randint(movement3+1, d3.shape[0])
    d4[movement4] = -noiseSize

    d1 = d1 + np.random.normal(0, 0.1, length)
    d2 = d2 + np.random.normal(0, 0.1, length)
    d3 = d3 + np.random.normal(0, 0.1, length)
    d4 = d4 + np.random.normal(0, 0.1, length)
    return np.array([d1, d4, d3, d2])

def generateClass4(length = 300, num=100, classno = 1, noiseSize=10):
    count = 0
    val = []
    while True:
        if count == num:
            return np.array(val), [classno for i in range(num)]
        else:
            try:
                val.append(generateSample4(length, noiseSize))
                count += 1
            except:
                continue


def generateSYN1(sizeTrainClass = 100, sizeTestClass = 20, length=200):
    x = np.arange(length)
    Fs = 100
    f = np.random.choice(2) # frequency
    weight = np.random.choice(2) # weight factor
    a = np.sin(2 * np.pi * 2 * (x) / Fs) * 1 + 1

    comp1 = np.zeros(length)
    comp1[150:] = a[37:87]
    comp2 = np.zeros(length)
    comp2[150:] = -a[37:87]
    comp3 = np.zeros(length)
    comp3[0:50] = a[37:87]
    comp4 = np.zeros(length)
    comp4[0:50] = -a[37:87]

    c1X, c1y = generateClass(comp3, comp2, num = sizeTrainClass, classno = 'a')
    c2X, c2y = generateClass(comp3, comp1, num = sizeTrainClass, classno = 'b')
    c3X, c3y = generateClass(comp4, comp2, num = sizeTrainClass, classno = 'c') 
    c4X, c4y = generateClass(comp4, comp1, num = sizeTrainClass, classno = 'd')   

    c1Xt, c1yt = generateClass(comp3, comp2, num=sizeTestClass, classno = 'a')
    c2Xt, c2yt = generateClass(comp3, comp1, num=sizeTestClass, classno = 'b')
    c3Xt, c3yt = generateClass(comp4, comp2, num=sizeTestClass, classno = 'c') 
    c4Xt, c4yt = generateClass(comp4, comp1, num=sizeTestClass, classno = 'd')   

    data = {}
    data['TRAIN'] = {}
    data['TEST'] = {}
    data['TRAIN']['X'] = np.concatenate([c1X, c2X, c3X, c4X])
    data['TRAIN']['y'] = np.concatenate([c1y, c2y, c3y, c4y])
    data['TEST']['X'] = np.concatenate([c1Xt, c2Xt, c3Xt, c4Xt])
    data['TEST']['y'] = np.concatenate([c1yt, c2yt, c3yt, c4yt])

    return data


def generateSYN2(noiseSize = 10, sizeTrainClass = 100, sizeTestClass = 20, length=300):

    c1X1, c1y1 =  generateClassTwoDIm(length=length, num=sizeTrainClass, classno = 'a', noiseSize= noiseSize)
    c2X1, c2y1 =  generateClassTwoDIm2(length=length, num=sizeTrainClass, classno = 'b', noiseSize= noiseSize)
    c1Xt1, c1yt1 =  generateClassTwoDIm(length=length, num=sizeTestClass, classno = 'a', noiseSize= noiseSize)
    c2Xt1, c2yt1 =  generateClassTwoDIm2(length=length, num=sizeTestClass, classno = 'b', noiseSize= noiseSize)

    data = {}
    data['TRAIN'] = {}
    data['TEST'] = {}

    data['TRAIN']['X'] = np.concatenate([c1X1, c2X1])
    data['TRAIN']['y'] = np.concatenate([c1y1, c2y1])
    data['TEST']['X'] = np.concatenate([c1Xt1, c2Xt1])
    data['TEST']['y'] = np.concatenate([c1yt1, c2yt1])

    return data

def generateSYN3(noiseSize = 10, sizeTrainClass = 100, sizeTestClass = 20, length=300):
    c1X2, c1y2 = generateClass2(length=length, num=sizeTrainClass, classno = 'a', noiseSize = noiseSize)
    c2X2, c2y2 = generateClass3(length=length, num=sizeTrainClass,  classno = 'b', noiseSize = noiseSize)
    c3X2, c3y2 = generateClass4(length=length, num=sizeTrainClass,  classno = 'c', noiseSize = noiseSize)
    c1Xt2, c1yt2 = generateClass2(length=length, num=sizeTestClass, classno = 'a', noiseSize = noiseSize)
    c2Xt2, c2yt2 = generateClass3(length=length, num=sizeTestClass, classno = 'b', noiseSize = noiseSize)
    c3Xt2, c3yt2 = generateClass4(length=length, num=sizeTestClass, classno = 'c', noiseSize = noiseSize)

    data = {}
    data['TRAIN'] = {}
    data['TEST'] = {}

    data['TRAIN']['X'] = np.concatenate([c1X2, c2X2, c3X2])
    data['TRAIN']['y'] = np.concatenate([c1y2, c2y2, c3y2])
    data['TEST']['X'] = np.concatenate([c1Xt2, c2Xt2, c3Xt2])
    data['TEST']['y'] = np.concatenate([c1yt2, c2yt2, c3yt2])

    return data
