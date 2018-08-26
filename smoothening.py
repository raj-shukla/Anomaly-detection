import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from itertools import groupby
from operator import itemgetter
import readData
np.set_printoptions(threshold=np.nan)

days = readData.days
flow = np.array(readData.flow)
flowList = np.array(readData.flowList)
time = np.array(readData.time)
postMile = np.array(readData.postMile)
lanes = np.array(readData.lanes)

flow = (flow - np.min(flow))/(np.max(flow) - np.min(flow))
flowArray = []

for i, val in enumerate(flow):
    flowArray.append(np.array(flow[i].reshape(24*12, 136)))
    

flowArray = np.asarray(flowArray)

point = 65
day = 5
flowAtPoint = flowArray[:, :, point:point+1]
flowAtPoint_1 = flowArray[:, :, point:point+1]

array = []

for i in range (0, len (flowAtPoint[5, 0:288, :]) - 5):
    array.append(np.mean(flowAtPoint[5, i:i+5, :]))


plt.plot(flowAtPoint[5, 0:288, :])
plt.show()

plt.plot(array)
plt.show()


