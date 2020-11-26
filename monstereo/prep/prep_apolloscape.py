import os
import sys
import time
import math
import copy
import json
import logging
from collections import defaultdict
import datetime
import numpy as np
import torch
from ..utils import correct_angle, normalize_hwl, pixel_to_camera, to_spherical #,append_cluster

from ..network.process import preprocess_monoloco, keypoints_dropout, clear_keypoints

from ..utils import K, KPS_MAPPING , APOLLO_CLUSTERS ,car_id2name, intrinsic_vec_to_mat, car_projection, pifpaf_info_extractor, keypoint_expander, keypoints_to_cad_model, set_logger, get_iou_matches





def extract_box_average(boxes_3d):
    boxes_np = np.array(boxes_3d)
    means = np.mean(boxes_np[:, 3:], axis=0)
    stds = np.std(boxes_np[:, 3:], axis=0)
    print(means)
    print(stds)




def bbox_gt_extract(bbox_3d, kk):
    zc = np.mean(bbox_3d[:,2])
    
    #take the top right corner and the bottom left corner of the bounding box in the 3D spac
    corners_3d = np.array([[np.min(bbox_3d[:,0]), np.min(bbox_3d[:,1]), zc], [np.max(bbox_3d[:,0]), np.max(bbox_3d[:,1]), zc] ])
    
    box_2d = []
    
    for xyz in corners_3d:
        xx, yy, zz = np.dot(kk, xyz)
        uu = xx / zz
        vv = yy / zz
        box_2d.append(uu)
        box_2d.append(vv)

    return box_2d
    
def append_cluster(dic_jo, phase, xx, ys, kps):
    """Append the annotation based on its distance"""

    for clst in APOLLO_CLUSTERS:
        try:
            if ys[3] <= int(clst):
                dic_jo[phase]['clst'][clst]['kps'].append(kps)
                dic_jo[phase]['clst'][clst]['X'].append(xx)
                dic_jo[phase]['clst'][clst]['Y'].append(ys)
                break
            
        except ValueError:
            dic_jo[phase]['clst'][clst]['kps'].append(kps)
            dic_jo[phase]['clst'][clst]['X'].append(xx)
            dic_jo[phase]['clst'][clst]['Y'].append(ys)




class PreprocessApolloscape:


    """Preprocess apolloscape dataset"""


    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
                          }
    dic_names = defaultdict(lambda: defaultdict(list))


    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def __init__(self, dir_ann, dataset, kps_3d = False, buffer=20, radius=200, dropout = 0, confidence = False, iou_min = 0.3):

        logging.basicConfig(level=logging.INFO)
        #self.logger = logging.getLogger(__name__)

        self.buffer = buffer

        #! To remove
        #self.radius = radius

        self.dropout =dropout
        
        self.kps_3d = kps_3d
        
        self.dir_ann = dir_ann

        self.confidence = confidence

        self.iou_min = iou_min

        dir_apollo = os.path.join('data', 'apolloscape')
        dir_out = os.path.join('data', 'arrays')

        assert os.path.exists(dir_apollo), "apollo directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M%S")[2:]


        try:
            process_mode = os.environ["process_mode"]
        except:
            process_mode = "NULL"

        identifier = '-apolloscape'

        kps_3d_id =""
        if self.kps_3d:
            identifier+="-kps_3d"
            kps_3d_id+="-kps_3d"

        name_out = 'ms-' + now_time + identifier+"-prep"+".txt"

        self.logger = set_logger(os.path.join('data', 'logs', name_out))
        self.logger.info("Preparation arguments: \nDir_ann: {} "
                         "\nprocess_mode : {} \nDropout images: {} \nConfidence keypoints: {} \nKeypoints 3D: {}".format(dir_ann, process_mode, dropout, confidence, self.kps_3d))


        self.path_joints = os.path.join(dir_out, 'joints-apolloscape-' + dataset + kps_3d_id + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-apolloscape-' + dataset + kps_3d_id + '-' + now_time + '.json')

        
        self.path  = os.path.join(dir_apollo, dataset)
        self.scenes, self.train_scenes, self.validation_scenes = factory(dataset, dir_apollo)



        
    def run(self):
        """
        Prepare arrays for training
        """
        cnt_scenes  = cnt_ann = 0
        start = time.time()

        occluded_keypoints=defaultdict(list)
        
        if self.dropout>0:
            dropouts = [0, self.dropout]
        else:
            dropouts = [0]

        for dropout in dropouts:
            
            if len(dropouts)>=2:
                self.logger.info("Generation of the inputs for a dropout of {}".format(dropout))

            for ii, scene in enumerate(self.scenes):
                
                if ii ==-100:
                    print("val_scenes",self.validation_scenes)

                if ii==(-10):
                    print("BREAK")
                    break

                cnt_scenes +=1 
                
                scene_id = scene.split("/")[-1].split(".")[0]
                camera_id = scene_id.split("_")[-1]
                car_poses = os.path.join(self.path, "car_poses", scene_id + ".json")
                
                #print(scene_id, self.train_scenes[:5])
                if scene_id+".jpg" in self.train_scenes:
                    phase = 'train'
                elif scene_id+".jpg" in self.validation_scenes:
                    phase ='val'
                else:    
                    print("phase name not in training or validation split")
                    continue
                    
                kk = K["Camera_"+camera_id]#intrinsic_vec_to_mat( K["Camera_"+camera_id])
                
                path_im = scene
                
                # Run IoU with pifpaf detections and save
                path_pif = os.path.join(self.dir_ann, scene_id+".jpg" + '.predictions.json')
                            
                if os.path.isfile(path_pif) and True:
                    boxes_gt_list, boxes_3d_list, kps_list, ys_list, car_model_list  = self.extract_ground_truth_pifpaf(car_poses,camera_id, scene_id, path_pif)

    
                self.dic_names[scene_id+".jpg"]['boxes'] = copy.deepcopy(list(boxes_gt_list))
                
                self.dic_names[scene_id+".jpg"]['car_model'] = copy.deepcopy(car_model_list)
                self.dic_names[scene_id+".jpg"]['K'] = copy.deepcopy(intrinsic_vec_to_mat(kk).tolist())

                ys_list_final = []
                #print("length keypoints list",len(kps_list))

                #print("length boxes gt list",len(boxes_gt_list))
                for kps, ys, boxes_gt, boxes_3d in zip(kps_list, ys_list, boxes_gt_list, boxes_3d_list):
                    
                    kps = [kps.transpose().tolist()]                    
                    

                    kps, length_keypoints, occ_kps = keypoints_dropout(kps, dropout, kps_3d = self.kps_3d)
                    occluded_keypoints[phase].append(occ_kps)
                    inp = preprocess_monoloco(kps,  intrinsic_vec_to_mat(kk).tolist(), confidence =self.confidence).view(-1).tolist()
                    
                    if self.kps_3d:
                        keypoints = clear_keypoints(kps, 3)
                        z = (keypoints[:, 2, :]).tolist()[0]
                        #print("Z", z)
                        ys = list(ys)
                        ys.append(z)
                        ys_list_final.append(ys)
                        #print("YS NEW", ys)
                        #raise ValueError

                    self.dic_jo[phase]['kps'].append(kps.tolist())
                    self.dic_jo[phase]['X'].append(list(inp))
                    self.dic_jo[phase]['Y'].append(ys)
                    self.dic_jo[phase]['names'].append(scene_id+".jpg")  # One image name for each annotation
                    self.dic_jo[phase]['boxes_3d'].append(list(boxes_3d))
                    self.dic_jo[phase]['K'].append(intrinsic_vec_to_mat(kk).tolist())
                    
                    append_cluster(self.dic_jo, phase, list(inp), ys, kps.tolist())
                    cnt_ann += 1
                    sys.stdout.write('\r' + 'Saved annotations {}'.format(cnt_ann) + '\t')

                
                self.dic_names[scene_id+".jpg"]['ys'] = copy.deepcopy(ys_list_final if self.kps_3d else ys_list)


        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        extract_box_average(self.dic_jo['train']['boxes_3d'])
        print("\nSaved {} annotations for {} scenes. Total time: {:.1f} minutes".format(cnt_ann, cnt_scenes, (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))    


        mean_val =   torch.mean(torch.Tensor(occluded_keypoints['val']))
        mean_train = torch.mean(torch.Tensor(occluded_keypoints['train']))
        std_val =    torch.std(torch.Tensor(occluded_keypoints['val']))
        std_train=   torch.std(torch.Tensor(occluded_keypoints['train']))

        self.logger.info("\nNumber of keypoints in the skeleton : {}\n"
                          "Val: mean occluded keypoints {:.4}; STD {:.4}\n"
                          "Train: mean occluded keypoints {:.4}; STD {:.4}\n".format(length_keypoints, mean_val, std_val, mean_train, std_train) )
        self.logger.info("\nOutput files:\n{}\n{}".format(self.path_names, self.path_joints))
        self.logger.info('-' * 120)
                 
    def extract_ground_truth_pifpaf(self, car_poses,camera_id, scene_id, path_pif):
        with open(car_poses) as json_file:
            data = json.load(json_file) #open the pose of the cars
        #print("NUMBER OF INSTANCES DETECTED BY PIFPAF", len(data))
        dic_vertices = {}
        dic_boxes = {}
        dic_poses = {}
        vertices_to_keypoints = {}
        dic_keypoints = {}
        dic_car_model = {}

        #Extract the boxes, vertices and the poses of each cars                   
        for car_index, car in enumerate(data):
            name_car = car_id2name[car['car_id']].name 
            car_model = os.path.join(self.path, "car_models_json",name_car+".json")

            intrinsic_matrix = intrinsic_vec_to_mat(K["Camera_"+camera_id])

            vertices_r, triangles, _ , w, l, h = car_projection(car_model, np.array([1,1,1]), T = np.array(car['pose']),  turn_over = True, bbox = False)
            vertices_2d = np.matmul(vertices_r,intrinsic_matrix.transpose()) # Projected vertices on the 2D plane
            x,y = (vertices_2d[:,0]/vertices_2d[:,2], vertices_2d[:,1]/vertices_2d[:,2])
            box_gt = [np.min(x), np.min(y), np.max(x), np.max(y)]
            #box_gt = bbox_gt_extract(bbox_3d, intrinsic_matrix)  # Take the overal bounding box in the 2D space
            
            dic_vertices[car_index] = vertices_2d
            dic_boxes[car_index] = [box_gt, w, l, h]
            dic_poses[car_index] = np.array(car['pose'])
            dic_car_model[car_index] = car_model

        new_keypoints = None
        
        #print("LENGTH OF THE EXTRACTED POSES",len(dic_poses))
        #print("LENGTH OF THE EXTRACTED vertices",len(dic_vertices))
        #  print("DIC_BOXES", dic_boxes)
                           
        if "sample" not in self.path:
                           
            #car_keypoints = os.path.join(self.path, "keypoints", scene_id)
            
            keypoints_list = []
            boxes_gt_list = []  # Countain the the 2D bounding box of the vehicles
            boxes_3d_list = []
            ys_list = []     
            car_model_list = []          
            
            keypoints_pifpaf = pifpaf_info_extractor(path_pif)
            #print("NUMBER OF KEYPOINTS of PIFPAF", len(keypoints_pifpaf))
            #print("MAME OF THE FILE", scene_id)
            #Compute the similarity between each set of car models in the 3D space and the set of keypoints from in the 2D space

            #? MAtch the 2d keypoints of the dataset with the keypoints from pifpaf
            boxes_kps=[]
            for index_keypoints, keypoints in enumerate(keypoints_pifpaf):
                
                dic_keypoints[index_keypoints] = np.array(keypoints)
                kps = keypoints[keypoints[:,0]>0][:, 1:]
                if len(kps)>1:
                    conf = np.sum(keypoints[:,0])
                    box_kp = [min(kps[:,0]), min(kps[:,1]),max(kps[:,0]), max(kps[:,1]), conf ]
                    boxes_kps.append(box_kp)

            boxes_gt=[]
            for i, vertices_2d in dic_vertices.items():
                x,y = (vertices_2d[:,0]/vertices_2d[:,2], vertices_2d[:,1]/vertices_2d[:,2])
                #print(x,y,vertices_2d[:,2] )
                boxes_gt.append([min(x),min(y), max(x), max(y)])
                
            matches = get_iou_matches(boxes_kps, boxes_gt, iou_min=self.iou_min)

            for match in matches:
                index_keypoint = match[0]
                index_vertice = match[1]
                
                vertices_to_keypoints[index_vertice] = [index_keypoint, None]
            

            #? With the keypoints now matched, extract the depth of the leypoints of the ground-truth and apply them to the keypoints
            #print("LENGTH OF VERTICES TO KEYPOINTS", len(vertices_to_keypoints))
            for index_cad, (index_keypoints, count) in vertices_to_keypoints.items()   :

                #if (index_cad != -1 and count >=1):

                keypoints = dic_keypoints[index_keypoints]
                vertices_2d = dic_vertices[index_cad]
                
                new_keypoints = keypoint_expander(vertices_2d, keypoints, self.buffer ,self.kps_3d)

                if self.kps_3d :
                    keypoints_list.append( new_keypoints[:,[1,2,3,0]])
                else:
                    keypoints_list.append( new_keypoints[:,[1,2,0]])
                                    
                boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                
                
                w, l, h = dic_boxes[index_cad][1:]
                pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation
                
                boxes_3d_list.append([xc, yc, zc, w, l, h])
                
                yaw = yaw%np.pi*2
                if yaw > np.pi:
                    yaw = yaw - 2*np.pi
                elif yaw < -np.pi:
                    yaw = yaw + np.pi*2
                
                sin, cos, _ = correct_angle(yaw, [xc, yc, zc])
                
                if True :
                    rtp = to_spherical([xc, yc, zc])
                    r, theta, psi = rtp # With r =d = np.linalg.norm([xc,yc,zc]) -> conversion to spherical coordinates 
                    #print("THETA, PSI, R", theta, psi, r)
                    ys_list.append([theta, psi, zc, r, h, w, l, sin, cos, yaw])
                else:
                    ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, sin, cos, yaw])
                
                
                car_model_list.append(dic_car_model[index_cad])
                
            #print("LENGTH OF YS_LIST", len(ys_list))     
            return boxes_gt_list, boxes_3d_list, keypoints_list, ys_list, car_model_list
        
        
        else:
            print("HERE")
            for index_cad, _ in dic_vertices.items() :
                
                boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                    
                w, l, h = dic_boxes[index_cad][1:]
                pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation

                boxes_3d_list.append([xc, yc, zc, w, l, h])
                yaw = yaw%np.pi

                ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, np.sin(yaw), np.cos(yaw), yaw])
            
            return boxes_gt_list, boxes_3d_list, None, ys_list
            
    
            
def factory(dataset, dir_apollo):
    """Define dataset type and split training and validation"""

    assert dataset in ['train', '3d_car_instance_sample']

    path = os.path.join(dir_apollo, dataset)
    
    if dataset == 'train':
        
        with open(os.path.join(path, "split", "train-list.txt"), "r") as file:
            train_scenes = file.read().splitlines()
        #scenes = [scene for scene in scenes if scene['token'] in train_scenes]
        with open(os.path.join(path, "split", "validation-list.txt"), "r") as file:
            validation_scenes = file.read().splitlines()
            
    elif dataset == '3d_car_instance_sample':
        with open(os.path.join(path, "split", "train-list.txt"), "r") as file:
            train_scenes = file.read().splitlines()
        #scenes = [scene for scene in scenes if scene['token'] in train_scenes]
        with open(os.path.join(path, "split", "validation-list.txt"), "r") as file:
            validation_scenes = file.read().splitlines()
    
    path_img = os.path.join(path, "images")
    scenes = [os.path.join(path_img, file) for file in os.listdir(path_img) if file.endswith(".jpg")]
    
    return scenes, train_scenes, validation_scenes

"""
def extract_ground_truth_pifpaf(self, car_poses,camera_id, scene_id, path_pif):
        with open(car_poses) as json_file:
            data = json.load(json_file) #open the pose of the cars
        #print("NUMBER OF INSTANCES DETECTED BY PIFPAF", len(data))
        dic_vertices = {}
        dic_boxes = {}
        dic_poses = {}
        vertices_to_keypoints = {}
        dic_keypoints = {}
        dic_car_model = {}

        #Extract the boxes, vertices and the poses of each cars                   
        for car_index, car in enumerate(data):
            name_car = car_id2name[car['car_id']].name 
            car_model = os.path.join(self.path, "car_models_json",name_car+".json")

            intrinsic_matrix = intrinsic_vec_to_mat(K["Camera_"+camera_id])

            vertices_r, triangles, bbox_3d , w, l, h = car_projection(car_model, np.array([1,1,1]), T = np.array(car['pose']),  turn_over = True, bbox = True
                                                           )
            vertices_2d = np.matmul(vertices_r,intrinsic_matrix.transpose()) # Projected vertices on the 2D plane
            
            box_gt = bbox_gt_extract(bbox_3d, intrinsic_matrix)  # Take the overal bounding box in the 2D space
            
            dic_vertices[car_index] = vertices_2d
            dic_boxes[car_index] = [box_gt, w, l, h]
            dic_poses[car_index] = np.array(car['pose'])
            dic_car_model[car_index] = car_model

        new_keypoints = None
        
        #print("LENGTH OF THE EXTRACTED POSES",len(dic_poses))
        #print("LENGTH OF THE EXTRACTED vertices",len(dic_vertices))
        #  print("DIC_BOXES", dic_boxes)
                           
        if "sample" not in self.path:
                           
            #car_keypoints = os.path.join(self.path, "keypoints", scene_id)
            
            keypoints_list = []
            boxes_gt_list = []  # Countain the the 2D bounding box of the vehicles
            boxes_3d_list = []
            ys_list = []     
            car_model_list = []          
            
            keypoints_pifpaf = pifpaf_info_extractor(path_pif)
            #print("NUMBER OF KEYPOINTS of PIFPAF", len(keypoints_pifpaf))
            #print("MAME OF THE FILE", scene_id)
            #Compute the similarity between each set of car models in the 3D space and the set of keypoints from in the 2D space
            for index_keypoints, keypoints in enumerate(keypoints_pifpaf):
                #print("INDEX KEYPOINTS", index_keypoints)
                dic_keypoints[index_keypoints] = keypoints

                k_t_c, index_cad, count = keypoints_to_cad_model_old(keypoints, dic_vertices, radius = self.radius)

                #print("INDEX_CAD, count", index_cad, count)
                #print(vertices_to_keypoints.keys() )
                if index_cad not in vertices_to_keypoints.keys() and count >= 7:
                    vertices_to_keypoints[index_cad] = [index_keypoints, count, k_t_c]
                elif index_cad in vertices_to_keypoints.keys()  and vertices_to_keypoints[index_cad][1] < count:
                    vertices_to_keypoints[index_cad] = [index_keypoints, count, k_t_c]
            
            #print("LENGTH OF VERTICES TO KEYPOINTS", len(vertices_to_keypoints))
            #print("CURRENTLY USED RADIUS", self.radius)
            for index_cad, (index_keypoints, count, k_t_c) in vertices_to_keypoints.items()   :

                if (index_cad != -1 and count >=1):

                    keypoints = dic_keypoints[index_keypoints]
                    vertices_2d = dic_vertices[index_cad]
                    
                    new_keypoints = keypoint_expander(vertices_2d, keypoints, self.buffer ,self.kps_3d)

                    if self.kps_3d :
                        keypoints_list.append( new_keypoints[:,[1,2,3,0]])
                    else:
                        keypoints_list.append( new_keypoints[:,[1,2,0]])
                                        
                    boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                    
                    
                    w, l, h = dic_boxes[index_cad][1:]
                    pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation
                    
                    boxes_3d_list.append([xc, yc, zc, w, l, h])
                    
                    # CHECK IF IT IS PI or 2PI
                    yaw = yaw%np.pi
                    
                    sin, cos, _ = correct_angle(yaw, [xc, yc, zc])
                    
                    if True :
                        rtp = to_spherical([xc, yc, zc])
                        r, theta, psi = rtp # With r =d = np.linalg.norm([xc,yc,zc]) -> conversion to spherical coordinates 
                        #print("THETA, PSI, R", theta, psi, r)
                        ys_list.append([theta, psi, zc, r, h, w, l, sin, cos, yaw])
                    else:
                        ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, sin, cos, yaw])
                    
                    
                    car_model_list.append(dic_car_model[index_cad])
            #print("LENGTH OF YS_LIST", len(ys_list))     
            return boxes_gt_list, boxes_3d_list, keypoints_list, ys_list, car_model_list
        
        
        else:
            print("HERE")
            for index_cad, _ in dic_vertices.items() :
                
                boxes_gt_list.append(dic_boxes[index_cad][0]) #2D corners of the bounding box
                    
                w, l, h = dic_boxes[index_cad][1:]
                pitch, yaw, roll, xc, yc, zc = dic_poses[index_cad] # Center position of the car and its orientation

                boxes_3d_list.append([xc, yc, zc, w, l, h])
                yaw = yaw%np.pi

                ys_list.append([xc, yc, zc, np.linalg.norm([xc, yc, zc]), h, w, l, np.sin(yaw), np.cos(yaw), yaw])
            
            return boxes_gt_list, boxes_3d_list, None, ys_list
"""