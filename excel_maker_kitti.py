#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
from collections import defaultdict


# In[2]:


files_set = defaultdict(lambda: defaultdict(str))

log_dir = "data/logs/"
index = 0
for file in sorted(os.listdir(log_dir)):
    if file.endswith(".txt"):
    
        if "prep" in file:
            if 'prep' not in files_set[index].keys():
                files_set[index]['prep'] = os.path.join(log_dir, file)
            else:
                print("alert")
                continue

        if "kitti" in file:
            if "train" not in files_set[index].keys():
                files_set[index]['train'] = os.path.join(log_dir, file)
            else:
                print("alert")
                continue

        if "eval" in file:
            if "eval" not in files_set[index].keys():
                files_set[index]['eval'] = os.path.join(log_dir, file)
            else:
                
                print("alert")
                continue

        if len(files_set[index].keys()) == 3:
            index+=1

if len(files_set[index].keys())!=3:
    print("not complete")
    files_set.pop(index)


# In[3]:


print("length of the dic", len(files_set))


# In[4]:


def parse_prep(file):

    
    try:
        with open(file, 'r') as file_object:
            
            lines = file_object.readlines()
            
            #print(lines, len(lines))
            
            for line in lines:

                
                if 'process_mode' in line :
                    process_mode = line.split()[-1]
                    
                if 'vehicles:' in line :
                    vehicles = line.split()[-1]
                    
                if 'Dropout' in line :
                    dropout = line.split()[-1]
                    
                if 'Dir_ann' in line :
                    dir_ann = line.split()[-1].split("/")[-1]
                    
                if 'Number of keypoints in the skeleton' in line:
                    num_kps = line.split()[-1]
                    
                if 'Val' in line:
                    occluded_val = line.split()[-1]
                    
                if 'Train' in line:
                    occluded_train = line.split()[-1]
        
        return process_mode, dropout, vehicles,num_kps, dir_ann,occluded_train, occluded_val
        
    except: 
        print("there was a problem with :", file)
        return None



def parse_train(file):
    
    try:
        with open(file, 'r') as file_object:
            
            lines = file_object.readlines()
            
            for line in lines:
                
                if 'epochs' in line :
                    epochs = line.split()[-1]
                    #print(process_mode)
                    
                if 'learning rate' in line :
                    l_r = line.split()[-1]
                    
                if 'input_size' in line :
                    input_size = line.split()[-1]
                    
                if 'output_size' in line :
                    output_size = line.split()[-1]
                    
                    
                if 'D:' in line and "2020" not in line:
                    D = line.split()[3]
                    bi = line.split()[7]
                    
                if 'X:' in line and "2020" not in line:
                    X = line.split()[1]
                    Y = line.split()[4]
                    
                if 'Ori:' in line and "2020" not in line:
                    Ori = line.split()[1]
                    
                if 'H:' in line and "2020" not in line:
                    H = line.split()[1]
                    W = line.split()[4]
                    L = line.split()[7]
                    
                      
                    
                if 'model saved' in line:
                    model = line.split()[-1]
        
        return epochs, l_r, input_size, output_size,model, D, bi, X, Y, Ori, H, W, L
        
    except: 
        print("there was a problem with :", file)
        return None
    
    
def parse_eval(file):
    #print(file)
    try:
        with open(file, 'r') as file_object:
            
            lines = file_object.readlines()
            
            for line in lines:

                #print(line)
                    
                if 'monoloco_pp' in line and "2020" not in line:
                    inf05 = line.split()[1]
                    inf1 = line.split()[2]
                    inf2 = line.split()[3]
                    
                    easy = line.split()[4]
                    easy_prob = line.split()[5][1:-2]
                    
                    medium = line.split()[6]
                    medium_prob = line.split()[7][1:-2]
                    
                    hard = line.split()[8]
                    hard_prob = line.split()[9][1:-2]
                    
                    all_sample = line.split()[10]
                    all_prob = line.split()[11][1:-2]
              
        
        return inf05, inf1, inf2, easy, easy_prob, medium, medium_prob, hard, hard_prob, all_sample, all_prob
        
    except: 
        print("there was a problem with :", file)
        return None
    


# In[5]:


full_study=[]


for index, file_set in files_set.items():    
    file_study = []
    if 'prep' in file_set.keys():
        process_mode, dropout, vehicles, num_kps, dir_ann,occluded_train, occluded_val = parse_prep(file_set['prep'])
        epochs, l_r, input_size, output_size,model, D, bi, X, Y, Ori, H, W, L = parse_train(file_set['train'])
        inf05, inf1, inf2, easy, easy_prob, medium, medium_prob, hard, hard_prob, all_sample, all_prob = parse_eval(file_set['eval'])
        
        file_study.append(dropout)
        file_study.append(vehicles)
        file_study.append(process_mode)
        file_study.append(dir_ann)
        file_study.append(num_kps)
        file_study.append(occluded_val)
        file_study.append(occluded_train)
        file_study.append(D)
        file_study.append(bi)
        file_study.append(X)
        file_study.append(Y)
        file_study.append(Ori)
        file_study.append(H)
        file_study.append(L)
        file_study.append(inf05)
        file_study.append(inf1)
        file_study.append(inf2)
        file_study.append(easy)
        file_study.append(easy_prob)
        file_study.append(medium)
        file_study.append(medium_prob)
        file_study.append(hard)
        file_study.append(hard_prob)
        file_study.append(all_sample)
        file_study.append(all_prob)
        
        
        full_study.append(file_study)


# In[ ]:


columns = ['dropout', 'vehicles', 'processing of the occluded keypoints', 'dataset', 'Number of keypoints', 'average number of occluded keypoints - VAL ', 'average number of occluded keypoints - TRAIN', 
          'D (m)', 'bi', 'X (cm)', 'Y (cm)', 'Ori (degrees)', 'H (cm)','W (cm)', 'L (cm)', '<0.5 m (%)', '<1 m (%)', '<2 m (%)'
          'easy (average distance error for the easy instance) [m]', 'percentage of recognized easy instances [%]', 'moderate (average distance error for the moderate instance) [m]', 'percentage of recognized moderate instances [%]',
          'hard (average distance error for the hard instance) [m]', 'percentage of recognized hard instances [%]', 'all (average distance error for all of the instances) [m]', 'percentage of recognized instances [%]']


# In[ ]:


df = pd.DataFrame(full_study, columns = columns)#, dtype = float) 


# In[ ]:


df.to_csv('process_summary.csv')

