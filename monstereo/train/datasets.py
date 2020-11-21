
import json
import torch

from torch.utils.data import Dataset


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

    def __init__(self, joints, phase, kps_3d = False):
        """
        Load inputs and outputs from the pickles files from gt joints, mask joints or both
        """
        assert(phase in ['train', 'val', 'test'])

        print("IN DATALOADER")
        with open(joints, 'r') as f:
            dic_jo = json.load(f)

        self.kps_3d = kps_3d


        
        # Define input and output for normal training and inference
        self.inputs_all = torch.tensor(dic_jo[phase]['X'])
        """print(len(self.inputs_all))
        print(len(self.inputs_all[0]))
        print(self.inputs_all[0][0])"""

        glob_list = []
        if self.kps_3d:
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

        #print("GLOABL LIST", glob_list[:5])

        self.outputs_all = torch.tensor(dic_jo[phase]['Y'] )
        print(len(self.outputs_all))
        print(len(self.outputs_all[0]))
        print(self.outputs_all[0][0])
        self.names_all = dic_jo[phase]['names']
        self.kps_all = torch.tensor(dic_jo[phase]['kps'])
        # Extract annotations divided in clusters
        self.dic_clst = dic_jo[phase]['clst']
        """print(self.dic_clst.keys())
        print(self.dic_clst[list(self.dic_clst.keys())[0]].keys())
        print(phase)"""
        #print(self.dic_clst['10']['kps'][0])

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

        return inputs, outputs, names, kps

    def get_cluster_annotations(self, clst):
        """Return normalized annotations corresponding to a certain cluster
        """
        if clst not in list(self.dic_clst.keys()):
            print("Cluster {} not in the data list :{}", clst, list(self.dic_clst.keys()))
            return None, None, None

        inputs = torch.tensor(self.dic_clst[clst]['X'])

        glob_list = []

        if self.kps_3d:
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
        count = len(self.dic_clst[clst]['Y'])

        return inputs, outputs, count
