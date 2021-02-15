#!/bin/bash -l


#? To be used on the izar cluster, rename the script with a .run extension and run it with sbatch
#SBATCH --nodes              1
#SBATCH --ntasks             1
#SBATCH --cpus-per-task      8
#SBATCH --partition gpu  
#SBATCH --qos gpu 
#SBATCH --gres gpu:1
#SBATCH --time 05:00:00


#? The format for the calls of ./train_eval.sh is the following:
#! ./train_eval.sh [usage on vehicles (0 for no, 1 for yes) ] [joint folder for the training] [dropout on the key-points] [joint files for the evaluation] [addiditonnal argument]

dropout='0.3'


export process_mode='mean'

    args='--confidence'
    #? Regular monoloco_pp network with the confidence term of each ke-ypoint added as part of the inputs

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

    args=''
    #? Regular monoloco_pp network
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

    args='--kps_3d'
    #? Regular monoloco_pp network extended to predict the depth of each 2D key-point.
    
    #source ./train_eval.sh 1 data/apollo-pifpaf/annotations ${dropout} apolloscape data/apollo-pifpaf/annotations "${args}"

    args='--confidence --lstm'
    #? LSTM implementation -> bad results
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"

    args='--transformer'
    #? self-attention based network

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"


    args='--transformer'
    #? self-attention based network extended to predict the depth component of each 2D key-point

    #source ./train_eval.sh 1 data/apollo-pifpaf/annotations ${dropout} apolloscape data/apollo-pifpaf/annotations "${args}"

    args='--transformer --scene_disp'
    #? self-attention based network, processing the inputs at the scene-level instead of the instance-level (experimental and not working well)


    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"
