
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from einops import rearrange, repeat

from ..network.architectures import SCENE_INSTANCE_SIZE

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
        # self.kps_all = torch.tensor(dic_jo[phase]['kps'])

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

        assert not (self.surround and self.scene_disp), "The surround techniuqe is not compatible with the batch sizes that akes into account the whole scene"

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
        EOS = repeat(torch.tensor([-10000]),'h -> h w', w = self.inputs_all.size(-1) )

        inputs_new = torch.zeros(len(np.unique(self.names_all)) , threshold,self.inputs_all.size(-1))
            
        output_new = torch.zeros(len(np.unique(self.names_all)) , threshold,self.outputs_all.size(-1))
        
        kps_new = torch.zeros(len(np.unique(self.names_all)), threshold,  self.kps_all.size(-2),self.kps_all.size(-1))
        
        #kps_v2 = torch.zeros(self.kps_all.size())
        
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
