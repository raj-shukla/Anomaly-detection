import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kurtosis, skew
from itertools import groupby
from operator import itemgetter
import threshhold
import analysis_data


depth = analysis_data.depth

threshhold = threshhold.threshhold



def FindAttack(tmp_attack_point, tmp_attack_type):
    group_attack = [list(g) for k, g in groupby(tmp_attack_type)]
    length_group = [len(i) for i in group_attack]
    for i in range(0, len(length_group) - 1):
        length_group[i+1] = length_group[i] + length_group[i+1]
    length_group = [i-1 for i in length_group]
    attack_type = [k for k, g in groupby(tmp_attack_type)]
    attack_point = [tmp_attack_point[i] for i in length_group]
    
    return attack_point, attack_type
    


def Analysis(point, day,  start, end, a_start, a_type):
    
    attack = [[] for i in range(0, 12)]
    attack_type_list = [[] for i in range(0, 12)]
    attack_points = [[] for i in range(0, 12)]
    attack_type = [[] for i in range(0, 12)]
    
    a_end = a_start + a_start/100
    tmp_array = np.copy(analysis_data.flowArray)
    flowWithAttack = analysis_data.flowArray[day, 0:288, point:point+1]
    
    if (a_type =="additive"): 
        flowWithAttack[start:end] = flowWithAttack[start:end] + random.uniform(a_start, a_end) 
    if (a_type == "deductive"):
        flowWithAttack[start:end] = flowWithAttack[start:end] - random.uniform(a_start, a_end) 
    if (a_type == "differential_1"):
        flowWithAttack[start:end/2] = flowWithAttack[start:end/2] + random.uniform(a_start, a_end) 
        flowWithAttack[end/2:end] = flowWithAttack[end/2:end] - random.uniform(a_start, a_end)
    if (a_type == "differential_2"):
        flowWithAttack[start:end/2] = flowWithAttack[start:end/2] - random.uniform(a_start, a_end) 
        flowWithAttack[end/2:end] = flowWithAttack[end/2:end] + random.uniform(a_start, a_end)    
    analysis_data.flowArray = np.copy(tmp_array)
         
    
    
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
        
        #print("###############")   
        #print(attack[i])
        #print(attack_type_list[i])
        #print (tmp_attack_point)
        #print (tmp_attack_type)
        print(attack_points[i])
        print(attack_type[i])
    
    return attack_points, attack_type 

def ProcessAttack(attack_points, attack_type):
    attack_sets = [attack_points.index(i) for i in attack_points if len(i) == 2]
    if (attack_sets == []):
        attack_sets = [attack_points.index(i) for i in attack_points if len(i) == 3]
    index = attack_sets[0]
    if (attack_sets == [])
        return None,None, None    
    return index, attack_points[index], attack_type[index] 
    
def FindAccuracy(true_value, detected_value, detected_type, index_list):

    false_start = 0
    false_end = 0
    diff_start = []
    diff_end = []
    for i in range (0, len(true_value)):
        if (detected_value[i] == None):
            false_start = false_start + 1
            false_end = false_end + 1
        else:
            if(len(detectected_value[i]) == 2):
                if ( abs(true_value[i][0] - detected_value[i][0]) > 10 ):
                    false_start = false_start + 1
                else:
                    diff_start.append(abs(true_value[i][0] - detected_value[i][0])) 
                if ( abs(true_value[i][1] - detected_value[i][1]) > 10 ):
                    false_end = false_end + 1
                else:
                    diff_end.append(abs(true_value[i][1] - detected_value[i][1])) 
            if (len(detectected_value[i]) == 3):
                if ((abs(true_value[i][0] - detected_value[i][0]) > 10) and (abs(true_value[i][0] - detected_value[i][1]) > 10)):
                    false_start = false_start + 1
                else:
                    diff_start.append(min(abs(true_value[i][0] - detected_value[i][0]),  abs(true_value[i][0] - detected_value[i][1]))) 
                if ( (abs(true_value[i][1] - detected_value[i][1]) > 10) and (abs(true_value[i][1] - detected_value[i][1]) > 10) ):
                    false_end = false_end + 1
                else:
                    diff_end.append(min(abs(true_value[i][1] - detected_value[i][1]),  abs(true_value[i][1] - detected_value[i][1]))) 
               
            
    return (false_start, false_end, diff_start, diff_end) 
        
    
attack_point_list = []
attack_type_list = []
true_value = []
detected_value = []
detected_type = []
index_list = []


for i in range(0, 100): 
    point = random.randint(0, 136)
    #point = 30
    day = random.randint(0, 7)
    #day = 5
    #flowAtPoint = flowArray[:, :, point:point+1] 
    #tmpFlowAtPoint = np.copy(flowAtPoint)
    start = random.randint(0, 200)
    end = start + random.randint(20, 80)
    #start = 100
    #end =120
    true_value.append([start, end])
    #print point
    print start
    print end
    #print (analysis_data.flowArray [day, start:end, point:point+1])
    attack_points_list, attack_type_list = Analysis (point, day, start, end, 0.20,  "additive")
    #flowAtPoint = np.copy(tmpFlowAtPoint)
    index, attack_points, attack_type = ProcessAttack(attack_points_list, attack_type_list)
    detected_value.append(attack_points)
    detected_type.append(attack_type)
    index_list.append(index)

statistics = FindAccuracy(true_value, detected_value, detected_type, index_list )

print (statistics)
print (index_list)

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

         

