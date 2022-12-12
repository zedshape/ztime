from ztime import syntheticGenerator
import numpy as np

data1 = syntheticGenerator.generateSYN1()
print(data1["TRAIN"]["X"].shape, np.unique(data1["TRAIN"]["y"]))
print(data1["TEST"]["X"].shape, np.unique(data1["TEST"]["y"]))

data2 = syntheticGenerator.generateSYN2()
print(data2["TRAIN"]["X"].shape, np.unique(data2["TRAIN"]["y"]))
print(data2["TEST"]["X"].shape, np.unique(data2["TEST"]["y"]))

data3 = syntheticGenerator.generateSYN3()
print(data3["TRAIN"]["X"].shape, np.unique(data3["TRAIN"]["y"]))
print(data3["TEST"]["X"].shape, np.unique(data3["TEST"]["y"]))