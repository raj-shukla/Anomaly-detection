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
flowAtPoint = flowArray[:, :, point:point+1]

array = flowArray[:, 100:150:, 5]

diff  = [[] for i in range(0, 12)]
attack = [[] for i in range(0, 12)]
attack_type_list = [[] for i in range(0, 12)]
attack_points = [[] for i in range(0, 12)]
attack_type = [[] for i in range(0, 12)]
meanFlow = []

for i in range(0, 7):
    print(np.mean(flowAtPoint[i:28:7, :, : ], axis=0))
    meanFlow.append(np.squeeze(np.mean(flowAtPoint[i:28:7, :, : ], axis=0)))

         
for i in range(0, 288 - len(diff)):
    for j in range(0, len(diff)):
        diff[j].append( np.absolute(np.mean(flowAtPoint[:, i:i+j+1, :]) - np.mean(flowAtPoint[:, i+j+1:i+j+1 + j+1, :])) )

     
threshhold = [max(i) for i in diff]

def Analysis(start):

    end = start + start/100
    flowWithAttack = flowAtPoint[5, 0:288, :] 
    flowWithAttack[150:200] = flowWithAttack[150:200] + 2*random.uniform(start, end)
    #flowWithAttack[50:90] = flowWithAttack[50:90] - 2*random.uniform(start, end)
    flowWithAttack[200:250] = flowWithAttack[200:250] - 2*random.uniform(start, end)
    
    
    for i in range(0, 288 -len(diff)):
        for j in range(0, len(diff)):
            if (abs( np.mean(flowWithAttack[i+j+1:i+j+1 + j+1]) - np.mean(flowWithAttack[i:i+j+1])) > threshhold[j]):
                attack[j].append(i+j+1)
                attack_type_list[j].append(np.sign(np.mean(flowWithAttack[i+j+1:i+j+1 + j+1]) - np.mean(flowWithAttack[i:i+j+1])))
    
    for i in range (0, len(attack)):
        for j, g in groupby(enumerate(attack[i]), lambda (k,x):k-x):
            tmpList = map(itemgetter(1), g)
            var = tmpList[len(tmpList)/2]
            attack_points[i].append(var)
        for j, g in groupby(attack_type_list[i]):
            attack_type[i].append(j)

tmpFlowAtPoint = np.copy(flowAtPoint)
Analysis (0.10)
flowAtPoint = np.copy(tmpFlowAtPoint)
print (attack_type)
print(attack_points)



         

