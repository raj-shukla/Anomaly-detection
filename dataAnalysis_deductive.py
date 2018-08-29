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
        
    
    return attack_points, attack_type 

def ProcessAttack(attack_points, attack_type):
    attack_sets = [attack_points.index(i) for i in attack_points if len(i) == 2]
    if (attack_sets == []):
        attack_sets = [attack_points.index(i) for i in attack_points if len(i) == 3]
    
    if (attack_sets ==  []):
        return None,None, None
    else:
        index = attack_sets[0]    
        return index, attack_points[index], attack_type[index] 
    
def FindAccuracy(true_value, detected_value, detected_type, index_list, margin):

    false_start = 0
    false_end = 0
    false_start_end = 0
    attack_type = 0
    diff_start = []
    diff_end = []

    for i in range (0, len(true_value)):
        if (detected_value[i] == None):
            false_start = false_start + 1
            false_end = false_end + 1
            false_start_end = false_start_end + 1
        else:
            start = False
            if(len(detected_value[i]) == 2):
                if (detected_type[i] !=  [-1, 1]):
                    attack_type = attack_type + 1
                if ( abs(true_value[i][0] - detected_value[i][0]) > margin ):
                    false_start = false_start + 1
                    start = True
                else:
                    diff_start.append(abs(true_value[i][0] - detected_value[i][0])) 
                if ( abs(true_value[i][1] - detected_value[i][1]) > margin ):
                    false_end = false_end + 1
                    if(start == True):
                        false_start_end = false_start_end + 1
                else:
                    diff_end.append(abs(true_value[i][1] - detected_value[i][1])) 
            if (len(detected_value[i]) == 3):
                if ((abs(true_value[i][0] - detected_value[i][0]) > margin) and (abs(true_value[i][0] - detected_value[i][1]) > margin)):
                    false_start = false_start + 1
                    start = True
                else:
                    minimum = min(abs(true_value[i][0] - detected_value[i][0]),  abs(true_value[i][0] - detected_value[i][1]))
                    diff_start.append(minimum) 
                    if (abs(true_value[i][0] - detected_value[i][0]) < abs(true_value[i][0] - detected_value[i][1])):
                        index = 0
                    else:
                        index = 1
                    if (detected_type[i][index] != -1):
                        attack_type = attack_type + 1
                if ( (abs(true_value[i][1] - detected_value[i][1]) > margin) and (abs(true_value[i][1] - detected_value[i][2]) > margin) ):
                    false_end = false_end + 1
                    if (start == True):
                        false_start_end = false_start_end + 1
                else:
                    minimum = min(abs(true_value[i][1] - detected_value[i][1]),  abs(true_value[i][1] - detected_value[i][1]))
                    diff_end.append(minimum) 
               
            
    return (false_start, false_end, false_start_end, np.mean(diff_start), np.mean(diff_end), attack_type) 
        
    
attack_point_list = []
attack_type_list = []
true_value = []
detected_value = []
detected_type = []
index_list = []
statistics = []
margin = 0

for i in range(0, 10):
    margin = margin + 10
    a_start = 0.20
    for j in range (0, 1):
        #a_start = a_start + 0.05
        for k in range(0, 1000): 
            point = random.randint(0, 136)
            day = random.randint(0, 7)
            start = random.randint(0, 200)
            end = start + random.randint(20, 80)
            true_value.append([start, end])
            attack_points_list, attack_type_list = Analysis (point, day, start, end, a_start,  "deductive")
            index, attack_points, attack_type = ProcessAttack(attack_points_list, attack_type_list)
            detected_value.append(attack_points)
            detected_type.append(attack_type)
            index_list.append(index)
        statistics.append(FindAccuracy(true_value, detected_value, detected_type, index_list, margin))
        true_value = []
        attack_points_list = []
        attack_type_lists = []
        detected_value = []
        detected_type = []
        index_list = []

for i in range(0, len(statistics)):
    print (statistics[i])

         

