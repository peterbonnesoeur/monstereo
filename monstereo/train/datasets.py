
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from einops import rearrange, repeat

from collections import defaultdict


from ..utils import get_iou_matrix
from ..network.architectures import SCENE_INSTANCE_SIZE, SCENE_LINE

class ActivityDataset(Dataset):
    """
    Dataloader for activity dataset
    """

    def __init__(self, joints, phase):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        assert(phase in ['train', 'val', 'test'])

        with open(joints, 'r') as f:
            dic_jo = json.load(f)

        # Define input and output for normal training and inference
        self.inputs_all = torch.tensor(dic_jo[phase]['X'])
        self.outputs_all = torch.tensor(dic_jo[phase]['Y']).view(-1, 1)

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        # kps = self.kps_all[idx, :]
        return inputs, outputs


class KeypointsDataset(Dataset):
    """
    Dataloader from nuscenes or kitti datasets
    """

    def __init__(self, joints, phase, kps_3d = False, transformer = False, surround = False, scene_disp = False):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        assert(phase in ['train', 'val', 'test'])

        print("IN DATALOADER")
        with open(joints, 'r') as f:
            dic_jo = json.load(f)

        self.kps_3d = kps_3d
        self.transformer = transformer
        self.surround = surround
        self.scene_disp = scene_disp
        
        # Define input and output for normal training and inference
        #TODO : perform additionnal tests on the surround argument 
        if self.surround : 
            self.envs_all = torch.tensor(dic_jo[phase]['env'])

        
        self.inputs_all = torch.tensor(dic_jo[phase]['X'])

        glob_list = []

        #? A different formatting was used for the kps_3D formatting
        # This step is just there to unfold the array befor being converted into a proper tensor
        if type(dic_jo[phase]['Y']) is list:
            for car_object in dic_jo[phase]['Y']:

                local_list=[]
                for item in car_object:
                    if isinstance(item, list):
                        for kp in item:
                            local_list.append(kp)
                    else:
                        local_list.append(item)
                glob_list.append(local_list)
            dic_jo[phase]['Y'] = glob_list


        self.outputs_all = torch.tensor(dic_jo[phase]['Y'] )
        self.names_all = dic_jo[phase]['names']
        self.kps_all = torch.tensor(dic_jo[phase]['kps'])

        if self.scene_disp:
            self.scene_disposition_dataset()

        self.dic_clst = dic_jo[phase]['clst']

    def __len__(self):
        """
        :return: number of samples (m)
        """
        return self.inputs_all.shape[0]

    def __getitem__(self, idx):
        """
        Reading the tensors when required. E.g. Retrieving one element or one batch at a time
        :param idx: corresponding to m
        """
        inputs = self.inputs_all[idx, :]
        outputs = self.outputs_all[idx]
        names = self.names_all[idx]
        kps = self.kps_all[idx, :]

        assert not (self.surround and self.scene_disp), "The surround technique is not compatible with the batch sizes that akes into account the whole scene"

        if self.surround:
            envs = self.envs_all[idx, :]
        else:
            envs = self.inputs_all[idx, :]

        return inputs, outputs, names, kps, envs

 
    def scene_disposition_dataset(self):
        
        #? in  order to perofmr a scene level transfromer, we need to create arrays with a fixed number of instances. 
        #! In our case, this number of instances is SCENE_INSTANCE_SIZE and is defined in network/architecture
        threshold = SCENE_INSTANCE_SIZE

        #? Test value to indicate the end of a sequence, or in our case, the end of the sequence of instances
        #! In practice, we do not use this value since it leads to worse results
        #EOS = repeat(torch.tensor([-10000]),'h -> h w', w = self.inputs_all.size(-1) )

        inputs_new = torch.zeros(len(np.unique(self.names_all)) , threshold,self.inputs_all.size(-1))
            
        output_new = torch.zeros(len(np.unique(self.names_all)) , threshold,self.outputs_all.size(-1))
        
        kps_new = torch.zeros(len(np.unique(self.names_all)), threshold,  self.kps_all.size(-2),self.kps_all.size(-1))
        
        
        old_name = None
        name_index = 0
        instance_index = 0
        
        print(np.argsort(self.names_all), np.sort(self.names_all))

        for i, index in enumerate(np.argsort(self.names_all)):
            if i == 0:
                old_name = self.names_all[index]
        
            #? If there are more instances on a scene than the maximum size of the array, the inputs are discarded.
            if instance_index >= threshold and old_name == self.names_all[index]:
                print("Too many instances in the images", self.names_all[index])
                pass
                

            #? If the old name is different from the new name in the list
            elif old_name != self.names_all[index]:
                #! In practice, do not use this value since it leads to worse results
                #if instance_index<threshold:
                #    inputs_new[name_index,instance_index,: ] = EOS
                instance_index = 0

                if old_name is not None:
                    name_index+=1          
                old_name = self.names_all[index]
                
                
                inputs_new[name_index,instance_index,: ] = self.inputs_all[index]
                output_new[name_index,instance_index,: ] = self.outputs_all[index]
                kps_new[name_index,instance_index,: ] = self.kps_all[index]
                
                instance_index+=1
                
                
            else:
                inputs_new[name_index,instance_index,: ] = self.inputs_all[index]
                output_new[name_index,instance_index,: ] = self.outputs_all[index]
                kps_new[name_index,instance_index,: ] = self.kps_all[index]
                
                instance_index+=1

        
        self.outputs_all = output_new
        self.inputs_all = inputs_new
        self.kps_all = kps_new
        self.names_all = np.unique(np.sort(self.names_all))

        if SCENE_LINE:
            self.line_scene_placement()
        

    def line_scene_placement(self):
            threshold = SCENE_INSTANCE_SIZE
                    
            inputs_new = torch.zeros(len(self.names_all)*threshold , threshold,self.inputs_all.size(-1))
                
            outputs_new = torch.zeros(len(self.names_all)*threshold , threshold,self.outputs_all.size(-1))
            kps_new = torch.zeros(len(self.names_all)*threshold, threshold,  self.kps_all.size(-2),self.kps_all.size(-1))
            
            names_new = []
            
            instance_index = 0
            
            for inputs, outputs, kps, names in zip(self.inputs_all, self.outputs_all, self.kps_all, self.names_all):
                mask = torch.sum(inputs, dim = 1) != 0
                
                if torch.sum(mask) >1:
        
                    offset = 0.2
                    kps_gen = rearrange(inputs, 'b (n d) -> b d n', d = 3)
                    box = rearrange(torch.stack((torch.min(kps_gen[mask][:,0,:], dim = -1)[0]-offset, torch.min(kps_gen[mask][:,1,:], dim = -1)[0]*0, torch.max(kps_gen[mask][:,0,:], dim = -1)[0]+offset, torch.max(kps_gen[mask][:,1,:], dim = -1)[0]*10000)), "b n -> n b")
                    #box = rearrange(torch.stack((torch.min(kps[mask][:,0,:], dim = -1)[0]-offset, torch.min(kps[mask][:,1,:], dim = -1)[0]-offset, torch.max(kps[mask][:,0,:], dim = -1)[0]+offset, torch.max(kps[mask][:,1,:], dim = -1)[0]+offset)), "b n -> n b")

                    #box = rearrange(torch.stack((torch.min(kps[mask][:,0,:], dim = -1)[0]*0, torch.min(kps[mask][:,1,:], dim = -1)[0], torch.max(kps[mask][:,0,:], dim = -1)[0]*10000, torch.max(kps[mask][:,1,:], dim = -1)[0])), "b n -> n b")
                    
                
                    pre_matches = get_iou_matrix(box, box)
                    matches = []
                    for i, match in enumerate(pre_matches):
                        for j, item in enumerate(match):
                            if item>0:
                                matches.append((i, j))
                
                    dic_matches = defaultdict(list)
                    
                    for match in matches:
                        if match[0] == match[1]:
                            pass
                        else:
                            dic_matches[match[0]].append(match[1])
                            dic_matches[match[1]].append(match[0])
                            
                    initialised = list(dic_matches.keys())
                    for i in range(len(box)):
                        if len(dic_matches[i]) ==0 and i not in initialised :
                            
                            inputs_new[instance_index, 0] = inputs[i]
                            outputs_new[instance_index,0] = outputs[i]
                            kps_new[instance_index, 0] = kps[i]
                            names_new.append(names)
                                                    
                            instance_index+=1
                        else:
                            list_match = dic_matches[i]
                            dic_matches.pop(i)
                            test = False
                            flag = True
                            while flag:
                                flag = False
                                for item in list_match:
                                    if len(dic_matches[item])>0:
                                        flag = True
                                        for match in dic_matches[item]:
                                            list_match.append(match)
                                        dic_matches.pop(item)
                                        
                            for count, match in enumerate(np.unique(list_match)):
                                inputs_new[instance_index, count] = inputs[match]
                                outputs_new[instance_index,count] = outputs[ match]
                                kps_new[instance_index, count] = kps[ match]
                                
                                test = True
                                
                            if len(list_match)>0:
                                instance_index+=1
                                names_new.append(names)
                else:
                    
                    inputs_new[instance_index] = inputs
                    outputs_new[instance_index] = outputs
                    kps_new[instance_index] = kps
                    names_new.append(names)
                    instance_index+=1
                    
            self.outputs_all = outputs_new[:instance_index]
            self.inputs_all = inputs_new[:instance_index]
            self.kps_all = kps_new[:instance_index]
            self.names_all = names_new
    
    def get_cluster_annotations(self, clst):
        """Return normalized annotations corresponding to a certain cluster
        """
        if clst not in list(self.dic_clst.keys()):
            print("Cluster {} not in the data list :{}", clst, list(self.dic_clst.keys()))
            return None, None, None, None

        inputs = torch.tensor(self.dic_clst[clst]['X'])

        glob_list = []

        if type(self.dic_clst[clst]['Y']) is list:
            for car_object in self.dic_clst[clst]['Y']:

                local_list=[]
                for item in car_object:
                    if isinstance(item, list):
                        for kp in item:
                            local_list.append(kp)
                    else:
                        local_list.append(item)
                glob_list.append(local_list)
            self.dic_clst[clst]['Y'] = glob_list

        outputs = torch.tensor(self.dic_clst[clst]['Y']).float()
        
        if self.surround:
            envs = torch.tensor(self.dic_clst[clst]['env']).float()
        else:
            envs = inputs
        count = len(self.dic_clst[clst]['Y'])
        return inputs, outputs, count, envs
