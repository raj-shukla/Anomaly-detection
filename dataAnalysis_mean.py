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
day = 4
flowAtPoint = flowArray[:, :, point:point+1]

print np.shape(flowAtPoint)

diff  = [[] for i in range(0, 12)]
mean  = [[] for i in range(0, 7)]
attack = [[] for i in range(0, 12)]
attack_type_list = [[] for i in range(0, 12)]
attack_points = [[] for i in range(0, 12)]
attack_type = [[] for i in range(0, 12)]
variance = []
standard_deviation = []

for i in range (0, 7):
    for j in range (0, 288):
        mean[i].append(np.mean(flowAtPoint[i:28:7, j, :])) 
    variance.append(np.var(mean[i]))
    standard_deviation.append(np.std(mean[i]))

#for i in range (0, len(mean)):
    #plt.plot(mean[i])
    #plt.grid()
    #plt.show()


         
print variance
print standard_deviation

threshhold = np.mean(standard_deviation)
print threshhold


def Analysis(start):

    end = start + start/100
    flowWithAttack = flowAtPoint[day, 0:288, :] 
    flowWithAttack[150:200:2] = flowWithAttack[150:200:2] + random.uniform(start, end) 
    flowWithAttack[151:200:2] = flowWithAttack[151:200:2] + random.uniform(start, end)       
    
    for i in range(0, 288):
        for j in range(0, 1):
            print np.absolute(flowWithAttack[i] -  mean[day][i])
            if ( np.absolute(flowWithAttack[i] -  mean[day][i]) > threshhold):
                attack[j].append(i)
                attack_type_list[j].append(np.sign( flowWithAttack[i] -  mean[day][i] ))
   
tmpFlowAtPoint = np.copy(flowAtPoint)
Analysis (0.20)
flowAtPoint = np.copy(tmpFlowAtPoint)
print (attack_type)
print(attack_points)

for i in range(0, len(attack_type)):
    print("################")
    print(len(attack_type[i]))
    print(len(attack_points[i]))

print (attack)
for i in range(0, len(attack)):
    plt.scatter(attack[i], attack[i])
    plt.grid()
    plt.show()
        

