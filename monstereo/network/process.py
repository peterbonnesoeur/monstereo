
import json
import os

import numpy as np
import torch
import torchvision
import torch
import random
from einops import rearrange

from collections import defaultdict

from ..utils import get_keypoints, pixel_to_camera, to_cartesian, back_correct_angles, get_iou_matrix

BF = 0.54 * 721
z_min = 4
z_max = 60
D_MIN = BF / z_max
D_MAX = BF / z_min

#! CHANGE HERE
KPS_NUMBER_3D_KPS = 24

def preprocess_monstereo(keypoints, keypoints_r, kk, vehicles = False, confidence = False):
    """
    Combine left and right keypoints in all-vs-all settings
    """
    clusters = []
    inputs_l = preprocess_monoloco(keypoints, kk, confidence=confidence)
    inputs_r = preprocess_monoloco(keypoints_r, kk, confidence = confidence)
    if vehicles:
        inputs = torch.empty((0, 96)).to(inputs_l.device)
        if confidence:
            inputs = torch.empty((0, 24*3*2)).to(inputs_l.device)
    else:
        inputs = torch.empty((0, 68)).to(inputs_l.device)
        if confidence:
            inputs = torch.empty((0, 17*3*2)).to(inputs_l.device)
    for idx, inp_l in enumerate(inputs_l.split(1)):
        clst = 0
        # inp_l = torch.cat((inp_l, cat[:, idx:idx+1]), dim=1)
        for idx_r, inp_r in enumerate(inputs_r.split(1)):
            # if D_MIN < avg_disparities[idx_r] < D_MAX:  # Check the range of disparities
            inp_r = inputs_r[idx_r, :]
            inp = torch.cat((inp_l, inp_l - inp_r), dim=1)  # (1,68)
            inputs = torch.cat((inputs, inp), dim=0)
            clst += 1
        clusters.append(clst)
    return inputs, clusters



def preprocess_monoloco(keypoints, kk, zero_center=False, kps_3d = False, confidence = False):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 24)/(m,2,24)  or list [3,24]/[2,24]
    Outputs =  torch tensors of (m, 48)/(m,72) in meters normalized (z=1) and zero-centered using the center of the box
    
    or, if we have the confidences:
     Outputs =  torch tensors of (m, 72)/(m,96) in meters normalized (z=1) and zero-centered using the center of the box
    """

    #? The confidence parameters adds the confidence for the keypoints in the process.
    
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)


    keypoints = clear_keypoints(keypoints, 2)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)
    if zero_center:
        uv_center = get_keypoints(keypoints, mode='center')
        xy1_center = pixel_to_camera(uv_center, kk, 10)
        kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, N, 3) - (m, 1, 3)
    else:
        kps_norm = xy1_all
    
    kps_out = kps_norm[:, :, 0:2]

    if confidence:
        kps_out = torch.cat((kps_out, keypoints[:, -1, :].unsqueeze(-1)), dim=2)

    kps_out = kps_out.reshape(kps_norm.size()[0], -1)  # no contiguous for view


    return kps_out


def reorganise_scenes(array, condition= "ypos", refining = False, descending = False):  
        #? The objective of this function is to reorganise the instances for the scene and refining step depending on some factor
        
        if refining:
            return None
        else:
            
            array = rearrange(array[:,:], 'b (n d) -> b n d', d = 3)

            mask = array[:,:,-1] != 0
            
            if condition == "ypos":
                a = (torch.sum(array[:,:,1], dim = 1)/torch.sum(mask, dim = 1)).to(array.device)
                
            elif condition == "xpos":
                a = torch.sum(array[:,:,0], dim = 1)/torch.sum(mask, dim = 1).to(array.device)
                
            elif condition == "height":
                a = (torch.max(array[:,:,1], dim = 1)[0]- torch.min(array[:,:,1], dim = 1)[0])/torch.sum(mask, dim = 1).to(array.device)
                
            elif condition == "width":
                a = (torch.max(array[:,:,0], dim = 1)[0]- torch.min(array[:,:,0], dim = 1)[0])/torch.sum(mask, dim = 1).to(array.device)  

            elif condition == "kps_num":
                a = torch.sum(mask, dim = 1).to(array.device)
                
            elif condition == "confidence":
                a = torch.sum(array[:,:,-1], dim = 1)/torch.sum(mask, dim = 1).to(array.device)
            else:
                return torch.arange(0, array.size(0)).to(array.device)
            
            sorted_array, indices = torch.sort(a, descending = descending)
            # Remove the incorrect elements -> technique to not consider the padde elements. 
            new_mask = ((torch.isinf(sorted_array)+torch.isnan(sorted_array)) == False).to(array.device)
                
            return indices[new_mask]

def reorganise_lines(inputs, offset = 0.2, unique  = False):  
        #? Function used to fragment the scenes into several clusters. 
        #? Those clusters correspond to the superposition of the instances in a scene.
        #? For exemple, in the context of heavy traffic, the superposition of several vehicles in the depth will result in a cluster.
        #? This also allows to create an easier to link set of data for the scene level attention mechanism.
        #? The "line" reference is linked to the shape of the clusters that regroups in heavy traffic several instances in the same line.
        inputs_new = torch.zeros(inputs.size(1) , inputs.size(1),inputs.size(-1)).to(inputs.device)
        
        instance_index = 0
        
        mask = (torch.sum(inputs[0], dim = 1) != 0).to(inputs.device)
                        
        if sum(mask) >1:
            inputs = inputs[0]
            kp = rearrange(inputs, 'b (n d) -> b d n', d = 3)
            indices_matches = []
            
            x_min = torch.min(kp[mask][:,0,:], dim = -1)[0]
            y_min = torch.min(kp[mask][:,1,:], dim = -1)[0]
            x_max = torch.max(kp[mask][:,0,:], dim = -1)[0]
            y_max = torch.max(kp[mask][:,1,:], dim = -1)[0]
                    
            offset_x = torch.abs(x_max-x_min)*offset
            offset_y = torch.abs(y_max-y_min)*offset
                    
                    
            box = rearrange(torch.stack((x_min-offset_x, y_min-offset_y, x_max+offset_x, y_max+ offset_y)), "b n -> n b")

            #? other option to have the boxes extended indefinitely in the y axis. 
            #? This way, as long as vehicles are"alligned" in the y axis, they will be grouped together
            #box = rearrange(torch.stack((x_min-offset_x, torch.zeros(y_min.size()).to(y_min.device), x_max+offset_x, torch.ones(y_max.size()).to(y_min.device))), "b n -> n b")

            pre_matches = get_iou_matrix(box, box)
            matches = []
            #! detect all the matches happening between our boxes (of course, a box has an intersection with itself)
            for i, match in enumerate(pre_matches):
                for j, item in enumerate(match):
                    if item>0:
                        matches.append((i, j))
        
            #? this defaultdict is made to register the matches between the different boxes
            #? and perform a chain of matches
            dic_matches = defaultdict(list)
            
            for match in matches:
                if match[0] != match[1]:
                    #? we don't register the matches between a box and itself
                    dic_matches[match[0]].append(match[1])
                    dic_matches[match[1]].append(match[0])
                    
            initialised = list(dic_matches.keys())
            
            for i in range(len(box)):
                if (len(dic_matches[i]) ==0 and i not in initialised) or unique :
                    #? Only matches with itself
                    inputs_new[instance_index, 0] = inputs[i]
                    indices_matches.append(i)
                    instance_index+=1
                else:
                    #? chain of matches
                    list_match = dic_matches[i]
                    dic_matches.pop(i)
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
                        indices_matches.append(match)
                                                
                    if len(list_match)>0:
                        # ? only update the name once all the matches are processed
                        instance_index+=1

            return inputs_new[:instance_index], torch.Tensor(indices_matches)
        else:
            #? Only one instance in the image
            return inputs, torch.Tensor([0])


def clear_keypoints(keypoints, nb_dim = 2):
    #? Clear the value of the occluded keypoints (confidence = 0) and replace them 
    #? with pre-defined values (used in monoloco_pp and the self-attention mechanism at the scene level)

    #! To rewrite in the last version of the code
    
    try:
        process_mode = os.environ["process_mode"]
    except:
        process_mode = 'mean'

    if process_mode=='':
        return keypoints

    for i, kps in enumerate(keypoints):

        #ensure that the  values are stored in the GPU
        mean = keypoints[i, 0:nb_dim, (keypoints[i,nb_dim, :]>0) ].mean(dim = 1).to(keypoints.device) 
        mean[mean != mean] = 0      # set Nan to 0
        if process_mode == 'neg':
            #? Set the occluded keypoints to a negative value
            mean =(torch.ones(mean.size())*-10).to(keypoints.device) 

        elif process_mode == 'zero':
            #? Set the occluded keypoints to 0
            mean =(torch.ones(mean.size())*0).to(keypoints.device) 
        
        if process_mode =='zero' or process_mode == 'mean' or process_mode == 'neg':
            if (keypoints[i,nb_dim, :]<=0).sum() != 0: # BE SURE THAT THE CONFIDENCE IS NOT EQUAL TO 0
                keypoints[i, 0:nb_dim, (keypoints[i,nb_dim, :]<=0)] = torch.transpose(mean.repeat((keypoints[i,nb_dim, :]<=0).sum() , 1), 0, 1)

        if process_mode=='mean_std':
            #? Replace the occluded keypoints by a subset of "synthetic" keypoints according to a normal distribution
            
            std = keypoints[i, 0:nb_dim, (keypoints[i,nb_dim, :]>0) ].std(dim = 1).to(keypoints.device) 
            std[std != std] = 0      # set Nan to 0
        
            if (keypoints[i,nb_dim, :]<=0).sum() != 0:
                #?Generation of an array of sythetic keypoints
                for j in range(len(keypoints[i,nb_dim, :])):
                    if keypoints[i,nb_dim, j]<=0:
                        keypoints[i, 0:nb_dim, j] = torch.normal(mean = mean, std = std/2)

        
    return keypoints

def factory_for_gt(im_size, name=None, path_gt=None, verbose=True):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    try:
        with open(path_gt, 'r') as f:
            dic_names = json.load(f)
        if verbose:
            print('-' * 120 + "\nGround-truth file opened")
    except (FileNotFoundError, TypeError): 
        if verbose:
            print('-' * 120 + "\nGround-truth file not found")
        dic_names = {}

    try:
        kk = dic_names[name]['K']
        dic_gt = dic_names[name]
        if verbose:
            print("Matched ground-truth file!")
    except KeyError:
        dic_gt = None
        x_factor = im_size[0] / 1600
        y_factor = im_size[1] / 900
        pixel_factor = (x_factor + y_factor) / 2  # 1.7 for MOT
        # pixel_factor = 1
        if im_size[0] / im_size[1] > 2.5:
            kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        else:
            kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                  [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                  [0., 0., 1.]]  # nuScenes calibration
        if verbose:
            print("Using a standard calibration matrix...")

    return kk, dic_gt


def laplace_sampling(outputs, n_samples):

    torch.manual_seed(1)
    mu = outputs[:, 0]
    bi = torch.abs(outputs[:, 1])

    # Analytical
    # uu = np.random.uniform(low=-0.5, high=0.5, size=mu.shape[0])
    # xx = mu - bi * np.sign(uu) * np.log(1 - 2 * np.abs(uu))

    # Sampling
    cuda_check = outputs.is_cuda
    if cuda_check:
        get_device = outputs.get_device()
        device = torch.device(type="cuda", index=get_device)
    else:
        device = torch.device("cpu")

    laplace = torch.distributions.Laplace(mu, bi)
    xx = laplace.sample((n_samples,)).to(device)

    return xx


def unnormalize_bi(loc):
    """
    Unnormalize relative bi of a nunmpy array
    Input --> tensor of (m, 2)
    """
    assert loc.size()[1] == 2, "size of the output tensor should be (m, 2)"
    bi = torch.exp(loc[:, 1:2]) * loc[:, 0:1]

    return bi


def preprocess_mask(dir_ann, basename, mode='left'):

    dir_ann = os.path.join(os.path.split(dir_ann)[0], 'mask')
    if mode == 'left':
        path_ann = os.path.join(dir_ann, basename + '.json')
    elif mode == 'right':
        path_ann = os.path.join(dir_ann + '_right', basename + '.json')

    from ..utils import open_annotations
    dic = open_annotations(path_ann)
    if isinstance(dic, list):
        return [], []

    keypoints = []
    for kps in dic['keypoints']:
        kps = prepare_pif_kps(np.array(kps).reshape(51,).tolist())
        keypoints.append(kps)
    return dic['boxes'], keypoints


def preprocess_pifpaf(annotations, im_size=None, enlarge_boxes=True, min_conf=0.):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 1 if enlarge_boxes else 2  # Avoid enlarge boxes for social distancing

    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        if conf >= min_conf:
            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)
        #print("RESULTS", boxes, keypoints)
    return boxes, keypoints


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""
    #print("KPS",len(kps_in))
    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]
    #print("XXS",len(xxs))
    return [xxs, yys, ccs]


def image_transform(image):

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    return transforms(image)


def extract_outputs(outputs, tasks=(), kps_3d = False):
    """
    Extract the outputs for multi-task training and predictions
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'x': outputs[:, 0],
               'y': outputs[:, 1],
               'd': outputs[:, 2:4], # Contain the distance and uncertianty term (for the Laplacian Loss)
               'h': outputs[:, 4],
               'w': outputs[:, 5],
               'l': outputs[:, 6],
               'ori': outputs[:, 7:9],
               }

    if outputs.shape[1] == 10:
        dic_out['aux'] = outputs[:, 9:10]
    
    if kps_3d:
        
        kps_size = KPS_NUMBER_3D_KPS
        dic_out['z_kps'] = outputs[:,-kps_size:]
        if outputs.shape[1] == 10+kps_size:
            dic_out['aux'] = outputs[:, 9:10]

        if len(tasks)>1 and "z_kp0" in tasks:
            for i in range(kps_size):
               dic_out['z_kp'+str(i)] = outputs[:, -kps_size+i:]
        

    # Multi-task training
    if len(tasks) >= 1:
    
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]

    # Preprocess the tensor
    # AV_H, AV_W, AV_L, HWL_STD = 1.72, 0.75, 0.68, 0.1
    bi = unnormalize_bi(dic_out['d'])
    dic_out['bi'] = bi

    dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
    x = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='x')
    y = to_cartesian(outputs[:, 0:3].detach().cpu(), mode='y')
    d = dic_out['d'][:, 0:1]
    z = torch.sqrt(d**2 - x**2 - y**2)
    dic_out['xyzd'] = torch.cat((x, y, z, d), dim=1)
    dic_out.pop('d')
    dic_out.pop('x')
    dic_out.pop('y')
    dic_out['d'] = d

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])
    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry

    if outputs.shape[1] == 10:
        dic_out['aux'] = torch.sigmoid(dic_out['aux'])
    
    if outputs.shape[1] == 11 and kps_3d:
        dic_out['aux'] = torch.sigmoid(dic_out['aux'])
    return dic_out


def extract_labels_aux(labels, tasks=None):

    dic_gt_out = {'aux': labels[:, 0:1]}

    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_gt_out[task] for task in tasks]

    dic_gt_out = {key: el.detach().cpu() for key, el in dic_gt_out.items()}
    return dic_gt_out


def extract_labels(labels, tasks=None, kps_3d = False):

    dic_gt_out = {'x': labels[:, 0:1], 'y': labels[:, 1:2], 'z': labels[:, 2:3], 'd': labels[:, 3:4],
                  'h': labels[:, 4:5], 'w': labels[:, 5:6], 'l': labels[:, 6:7],
                  'ori': labels[:, 7:9], 'aux': labels[:, 10:11]}

    if kps_3d:
        kps_size = 24
        
        dic_gt_out['z_kps'] = labels[:, -kps_size:]
        #print("LABELS Z_KPS", dic_gt_out['z_kps'] )

        if tasks is not None and "z_kp0" in tasks:
                #for i, z_kp in enumerate(dic_gt_out['z_kps']):
                #    dic_gt_out['z_kp'+str(i)] = z_kp
                for i in range(kps_size):
                    dic_gt_out['z_kp'+str(i)] = labels[:, -kps_size+i:]

    if tasks is not None:
        try:

            assert isinstance(tasks, tuple), "tasks need to be a tuple"
            #raise KeyError
            return [dic_gt_out[task] for task in tasks]
        except KeyError:
            print("TASK LIST extract", tasks)
            print("Keypoints key list", dic_gt_out.keys())
            print("DIC_OUT", dic_gt_out)

    dic_gt_out = {key: el.detach().cpu() for key, el in dic_gt_out.items()}

    x = to_cartesian(labels[:, 0:3].detach().cpu(), mode='x')
    y = to_cartesian(labels[:, 0:3].detach().cpu(), mode='y')
    d = dic_gt_out['d'][:, 0:1]
    z = torch.sqrt(d**2 - x**2 - y**2)
    dic_gt_out['xyzd'] = torch.cat((x, y, z, d), dim=1)

    yaw_pred = torch.atan2(dic_gt_out['ori'][:, 0:1], dic_gt_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_gt_out['xyzd'][:, 0:3])
    dic_gt_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry

    return dic_gt_out


def cluster_outputs(outputs, clusters):
    """Cluster the outputs based on the number of right keypoints"""

    # Check for "no right keypoints" condition
    if clusters == 0:
        clusters = max(1, round(outputs.shape[0] / 2))

    assert outputs.shape[0] % clusters == 0, "Unexpected number of inputs"
    outputs = outputs.view(-1, clusters, outputs.shape[1])
    return outputs


def filter_outputs(outputs):
    """Extract a single output for each left keypoint"""

    # Max of auxiliary task
    val = outputs[:, :, -1]
    best_val, _ = val.max(dim=1, keepdim=True)
    mask = val >= best_val
    output = outputs[mask]  # broadcasting happens only if 3rd dim not present
    return output, mask


def extract_outputs_mono(outputs, tasks=None):
    """
    Extract the outputs for single di
    Inputs:
        tensor (m, 10) or (m,9) if monoloco
    Outputs:
         - if tasks are provided return ordered list of raw tensors
         - else return a dictionary with processed outputs
    """
    dic_out = {'xyz': outputs[:, 0:3], 'zb': outputs[:, 2:4],
               'h': outputs[:, 4:5], 'w': outputs[:, 5:6], 'l': outputs[:, 6:7], 'ori': outputs[:, 7:9]}

    # Multi-task training
    if tasks is not None:
        assert isinstance(tasks, tuple), "tasks need to be a tuple"
        return [dic_out[task] for task in tasks]

    # Preprocess the tensor
    bi = unnormalize_bi(dic_out['zb'])

    dic_out = {key: el.detach().cpu() for key, el in dic_out.items()}
    dd = torch.norm(dic_out['xyz'], p=2, dim=1).view(-1, 1)
    dic_out['xyzd'] = torch.cat((dic_out['xyz'], dd), dim=1)

    dic_out['d'], dic_out['bi'] = dd, bi

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])

    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry
    return dic_out


def keypoints_dropout(keypoints, dropout = 0 ,nb_dim =2, kps_3d = False):

    if kps_3d:
        nb_dim = 3
    else:
        nb_dim = 2

    length_keypoints = 0
    occluded_kps = []

    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)

    for i, _ in enumerate(keypoints):
        length_keypoints = len(keypoints[i,nb_dim, :])
        threshold = int(length_keypoints*(dropout))

        #print(length_keypoints, list(range(length_keypoints)))
        #print())
        kps_list = random.sample(list(range(length_keypoints)), length_keypoints)
        #print("KEYPOINT_LIST", kps_list)
        for j in kps_list:
            val = torch.rand(1)
            if val<dropout: #(keypoints[i,nb_dim, :]>0).sum() >= threshold and val<self.dropout: # BE SURE THAT THE CONFIDENCE IS NOT EQUAL TO 0
                keypoints[i, 0:nb_dim, j] = torch.tensor([-5]*nb_dim)
                keypoints[i, nb_dim, j] = torch.tensor(0)

        #print("NUM OCCLUDED ", (keypoints[i,nb_dim, :]<=0).sum() )
        occluded_kps.append((keypoints[i,nb_dim, :]<=0).sum())
    return keypoints, length_keypoints, occluded_kps

def dist_angle_array(keypoints_list, base_dim = 10):

    assert keypoints_list.size()[1]%3 == 0, "You need the confidence on this one"
    assert base_dim%2 == 0, "We want to have a vector for both the angles and distnces. For this, the base dim needs to be a multiple of 2"

    device = keypoints_list.device
    #print("MY DEVICE", device)
    keypoints_list = rearrange(keypoints_list, 'b (n d) -> b n d', d = 3)

    conf_masks = keypoints_list[:,:, -1]>0
    conf_masks = conf_masks.to(device)
    #print(conf_masks.device, keypoints_list.device )
    #print(conf_masks.size(), keypoints_list.size()  ,keypoints_list[0][conf_masks[0]].size())
    #print(keypoints_list[0], conf_masks[0], keypoints_list[0][conf_masks[0]])
    
    #? Here, we compute the mean values of each sets of keypoints and we only use the keypoints with a confidence > 0
    #means = []
    #for keypoints, conf_mask in zip(keypoints_list, conf_masks):
    #    keypoints=keypoints.to(device)
    #    conf_mask=conf_mask.to(device)
    #    a = keypoints
    #    a = a[conf_mask]
    #    means.append(torch.mean(a, dim =0)[:2])
    means = [torch.mean(keypoints[conf_mask], dim =0)[:2] for keypoints, conf_mask in zip(keypoints_list, conf_masks)]

    #print(means)
    
    means = torch.stack(means)
    #print(means.device)
    dists = []
    angles = []

    v = torch.Tensor([1,0]).to(device)
    for mean in means:
        #print(means,mean)
        #print(torch.norm(means-mean))
        #print("here",torch.norm(means-mean, dim = 1),list(torch.sort(torch.norm(means-mean, dim = 1)))[1:])
        dist, arg =  torch.sort(torch.norm(means-mean, dim = 1))
        #print(dist[1:], arg[1:])
        dists.append(dist[1:])
        args = arg[1:]
        
        #angles.append(np.array([np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v) for u in (means-mean)  ])[args] )
        #print([torch.dot(u,v)/torch.norm(u)/torch.norm(v) for u in (means-mean)  ])
        angles.append(torch.stack([torch.dot(u,v)/torch.norm(u)/torch.norm(v) for u in (means-mean)  ])[args] )
    
    dists = torch.stack(dists)
    angles = torch.stack(angles)
    #print(angles, dists)
        
    vec_dist = torch.zeros((len(keypoints_list), base_dim)).to(device)
    #print(vec_dist)
    for j in range(len(keypoints_list)):
        for i in range(min ( int(base_dim/2), len(angles[0]))):

            #print(i,j, dists[j])
            vec_dist[j, i*2] = dists[j, i]
            vec_dist[j,i*2+1] = angles[j, i]
    
    #print(vec_dist)
    return vec_dist