import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from itertools import groupby
from operator import itemgetter
import readData
import threshhold

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

point = 30
day = 5
flowAtPoint = flowArray[:, :, point:point+1]
print np.shape(flowAtPoint)

threshhold = threshhold.threshhold
attack = [[] for i in range(0, 12)]
attack_type_list = [[] for i in range(0, 12)]
attack_points = [[] for i in range(0, 12)]
attack_type = [[] for i in range(0, 12)]
depth = 12

def FindAttack(tmp_attack_point, tmp_attack_type):
    group_attack = [list(g) for k, g in groupby(tmp_attack_type)]
    length_group = [len(i) for i in group_attack]
    #print group_attack
    #print length_group
    for i in range(0, len(length_group) - 1):
        length_group[i+1] = length_group[i] + length_group[i+1]
        #print("################")
        #print(i)
        #print(length_group[i])
        #print(length_group[i+1])
    #print (length_group)
    length_group = [i-1 for i in length_group]
    #print(length_group)
    attack_type = [k for k, g in groupby(tmp_attack_type)]
    attack_point = [tmp_attack_point[i] for i in length_group]
    
    return attack_point, attack_type
    


def Analysis(start):
     

    end = start + start/100
    flowWithAttack = flowAtPoint[day, 0:288, :] 
    flowWithAttack[150:180] = flowWithAttack[150:180] + random.uniform(start, end) 
    #flowWithAttack[151:200:2] = flowWithAttack[151:200:2] + random.uniform(start, end)       
    
    for i in range(0, 288 - depth):
        for j in range(0, depth):
            if (abs( np.mean(flowWithAttack[i+j+1:i+j+1 + j+1]) - np.mean(flowWithAttack[i:i+j+1])) > threshhold[j]):
                attack[j].append(i+j+1)
                attack_type_list[j].append(np.sign(np.mean(flowWithAttack[i+j+1:i+j+1 + j+1]) - np.mean(flowWithAttack[i:i+j+1])))
    
    for i in range (0, len(attack)):
        tmp_attack_point = []
        tmp_attack_type = []
        for j, g in groupby(enumerate(attack[i]), lambda (k,x):k-x):
            tmpList = map(itemgetter(1), g)
            var_1 = tmpList[len(tmpList)/2]
            tmp_attack_point.append(var_1)
            var_2 = attack_type_list[i][attack[i].index(var_1)]
            tmp_attack_type.append(var_2)
        attack_points[i], attack_type[i] = FindAttack(tmp_attack_point, tmp_attack_type)
        
        print("###############")   
        print(attack[i])
        print(attack_type_list[i])
        print (tmp_attack_point)
        print (tmp_attack_type)
        print(attack_points[i])
        print(attack_type[i])
        

tmpFlowAtPoint = np.copy(flowAtPoint)
Analysis (0.20)
flowAtPoint = np.copy(tmpFlowAtPoint)
#print (attack_type)
#print(attack_points)

'''
plt.plot(flowAtPoint[day, 0:288, :])
plt.show()
for i in range(0, len(attack_type)):
    print("################")
    print(len(attack_type[i]))
    print(len(attack_points[i]))

for i in range(0, len(attack)):
    plt.scatter(attack[i], attack[i])
    plt.grid()
    plt.grid()
    plt.show()
'''

         

