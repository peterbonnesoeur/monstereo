
import math
import os
import json
import time
import logging
import random
import datetime

import torch
import numpy as np
import pandas as pd

from .trainer import Trainer
from ..eval import EvalKitti, GenerateKitti

from ..utils import set_logger

from collections import defaultdict



class HypTuning:

    def __init__(self, joints, epochs, monocular, dropout, multiplier=1, r_seed=7, vehicles=False, kps_3d = False, dataset = 'kitti', 
                confidence = False, transformer = False, lstm = False, scene_disp = False, scene_refine = False, dir_ann = None):
        """
        Initialize directories, load the data and parameters for the training
        """

        # Initialize Directories
        self.dataset = dataset
        self.vehicles = vehicles
        self.joints = joints
        self.monocular = monocular
        self.dropout = dropout
        self.num_epochs = epochs
        self.r_seed = r_seed

        self.confidence = confidence
        self.transformer = transformer
        self.lstm = lstm
        self.scene_disp = scene_disp
        self.dir_ann = dir_ann

        self.scene_refine = scene_refine
        if self.scene_refine:
            self.scene_disp = True

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M%S")[2:]
        name_out = 'ms-' + now_time+'-'+"hyperparameter_study"+".txt"
        self.logger = set_logger(os.path.join('data', 'logs', name_out))

        dir_out = os.path.join('data', 'models')
        dir_logs = os.path.join('data', 'logs')
        assert os.path.exists(dir_out), "Output directory not found"
        if not os.path.exists(dir_logs):
            os.makedirs(dir_logs)

        name_out = 'hyp-monoloco-' if monocular else 'hyp-ms-'

        self.path_log = os.path.join(dir_logs, name_out)
        self.path_model = os.path.join(dir_out, name_out)

        self.kps_3d = kps_3d 
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.dic_results = defaultdict(lambda: defaultdict(list))

        # Initialize grid of parameters
        random.seed(r_seed)
        np.random.seed(r_seed)
        self.sched_gamma = [1] * 10* multiplier

        aa = math.log(1.5, 10)
        bb = math.log(0.1, 10)
        log_lr_list = np.random.uniform(aa, bb, int(len(self.sched_gamma))).tolist()
        self.sched_gamma = [10 ** xx for xx in log_lr_list]
        random.shuffle(self.sched_gamma)

        self.logger.info("Sched gamma list {} of length {}".format( self.sched_gamma, len(self.sched_gamma)))

        steps = np.random.uniform(int(self.num_epochs/10), int(self.num_epochs*90/100), int(len(self.sched_gamma))).tolist()
        self.sched_step= [int(xx) for xx in steps]
        random.shuffle(self.sched_step)
        
        self.logger.info("Sched step list {} of length {}".format( self.sched_step, len(self.sched_step)))


        self.bs_list = [128, 128,  256, 256, 256, 512, 512, 512, 1024, 1024]* multiplier

        random.shuffle(self.bs_list)

        self.hidden_list = [512, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024] * multiplier
        random.shuffle(self.hidden_list)

        self.n_stage_list = [3]*10 * multiplier
        random.shuffle(self.n_stage_list)

        self.num_heads_list = [4]*10 * multiplier

        random.shuffle(self.num_heads_list)
        # Learning rate
        aa = math.log(0.00008, 10)
        bb = math.log(0.1, 10)
        log_lr_list = np.random.uniform(aa, bb, int(len(self.sched_gamma))).tolist()
        self.lr_list = [10 ** xx for xx in log_lr_list]



        self.logger.info("Lr list {} of length {}".format( self.lr_list, len(self.lr_list)))


    def train(self):
        """Train multiple times using log-space random search"""

        best_acc_val = 20
        dic_best = {}
        dic_err_best = {}
        start = time.time()
        cnt = 0
        for idx, lr in enumerate(self.lr_list):
            bs = self.bs_list[idx]
            sched_gamma = self.sched_gamma[idx]
            sched_step = self.sched_step[idx]
            hidden_size = self.hidden_list[idx]
            n_stage = self.n_stage_list[idx]
            num_heads = self.num_heads_list[idx]

            training = Trainer(joints=self.joints, epochs=self.num_epochs,
                               bs=bs, monocular=self.monocular, dropout=self.dropout, lr=lr, sched_step=sched_step,
                               sched_gamma=sched_gamma, hidden_size=hidden_size, n_stage=n_stage, num_heads =num_heads,
                               save=False, print_loss=False, r_seed=self.r_seed, vehicles=self.vehicles, kps_3d = self.kps_3d,
                               dataset = self.dataset, confidence= self.confidence, transformer = self.transformer, 
                            lstm = self.lstm, scene_disp =self.scene_disp, scene_refine = self.scene_refine)

            best_epoch = training.train()
            dic_err, model = training.evaluate()
            acc_val = dic_err['val']['all']['mean']
            cnt += 1
            self.logger.info("Combination number: {}".format(cnt))
            
            if self.dataset == 'kitti':

                self.logger.info("Sched_gamma {} \nSched_step {} \nHidden size : {} \nN stages {}\nLearning rate {}"
                .format(sched_gamma, sched_step, hidden_size, n_stage, lr))

                self.logger.info("Trainer(joints={}, epochs={},\n"
                                "bs={}, monocular={}, dropout={}, lr={}, sched_step={},\n"
                                "sched_gamma={}, hidden_size={}, n_stage={},\n n_heads={},\n"
                                "save=False, print_loss=False, r_seed={}, vehicles={}, kps_3d = {},\n"
                                "dataset = {}, confidence= {}, transformer = {}, \n"
                                "lstm = {}, scene_disp ={}, scene_refine = {} )\n"
                                .format(self.joints,self.num_epochs,bs, self.monocular, self.dropout, lr, sched_step,
                                sched_gamma, hidden_size, n_stage, num_heads, self.r_seed,self.vehicles, self.kps_3d, self.dataset, self.confidence,
                                self.transformer,  self.lstm, self.scene_disp, self.scene_refine))

                if self.monocular:
                    model_stereo = "nope"
                    model_mono = model
                else:
                    model_stereo = model
                    model_mono = "nope"


                kitti_txt = GenerateKitti(model_stereo, self.dir_ann, p_dropout=0, n_dropout=0,
                                          hidden_size=hidden_size, vehicles = self.vehicles, model_mono = model_mono,
                                          confidence = self.confidence, transformer = self.transformer, 
                                          lstm = self.lstm, scene_disp = self.scene_disp, scene_refine = self.scene_refine)

                kitti_txt.run()
                

                kitti_eval = EvalKitti(verbose=True, vehicles = self.vehicles, dir_ann=self.dir_ann, transformer=self.transformer, logger = self.logger)
                kitti_eval.run()

            
            self.dic_results[lr]["sched_step"].append(sched_step)
            self.dic_results[lr]["sched_gamma"].append(sched_gamma)
            self.dic_results[lr]["bs"].append(bs)
            self.dic_results[lr]["hidden_size"].append(hidden_size)
            self.dic_results[lr]["acc_val"].append(dic_err['val']['all']['d'])



            if acc_val < best_acc_val:
                dic_best['lr'] = lr
                dic_best['joints'] = self.joints
                dic_best['bs'] = bs
                dic_best['monocular'] = self.monocular
                dic_best['sched_gamma'] = sched_gamma
                dic_best['sched_step'] = sched_step
                dic_best['hidden_size'] = hidden_size
                dic_best['n_stage'] = n_stage
                dic_best['acc_val'] = dic_err['val']['all']['d']
                dic_best['best_epoch'] = best_epoch
                dic_best['random_seed'] = self.r_seed
                # dic_best['acc_test'] = dic_err['test']['all']['mean']

                dic_err_best = dic_err
                best_acc_val = acc_val
                model_best = model

        for lr in self.dic_results.keys():
            self.logger.info("HERE for the dataframe conversion")
            dataframe = pd.DataFrame.from_dict(self.dic_results[lr])

            type_inst = "vehicle" if self.vehicles else "human"
            name_out = "hyperparameter_"+self.dir_ann.split("/")[-1]+"_results_"+type_inst+"_lr_"+str(lr)+".csv"
            output_file = os.path.join('data', 'logs', name_out)
            dataframe.to_csv(output_file)


        # Save model and log<
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_model = self.path_model + now_time + '.pkl'
        torch.save(model_best.state_dict(), self.path_model)
        with open(self.path_log + now_time, 'w') as f:
            json.dump(dic_best, f)
        end = time.time()
        print('\n\n\n')
        self.logger.info(" Tried {} combinations".format(cnt))
        self.logger.info(" Total time for hyperparameters search: {:.2f} minutes".format((end - start) / 60))
        self.logger.info(" Best hyperparameters are:")
        for key, value in dic_best.items():
            self.logger.info(" {}: {}".format(key, value))

        
        self.logger.info("Accuracy in each cluster:")

        for key in ('10', '20', '30', '>30', 'all'):
            self.logger.info("Val: error in cluster {} = {} ".format(key, dic_err_best['val'][key]['d']))
        self.logger.info("Final accuracy Val: {:.2f}".format(dic_best['acc_val']))
        self.logger.info("\nSaved the model: {}".format(self.path_model))
        print(self.path_model)
        print(dic_best['hidden_size'])