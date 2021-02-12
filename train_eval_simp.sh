#!/bin/bash -l

#SBATCH --nodes              1
#SBATCH --ntasks             1
#SBATCH --cpus-per-task      8
#SBATCH --partition gpu  
#SBATCH --qos gpu 
#SBATCH --gres gpu:1
#SBATCH --time 05:00:00


#? The format for the calls of ./train_eval.sh is the following:
#! ./train_eval.sh [usage on vehicles (0 for no, 1 for yes) ] [joint files for the training] [dropout on the key-points] [joint files for the evaluation] [addiditonnal argument]

dropout='0.3'

args='--confidence'

export process_mode='NULL_ISNT_IT'
    #source ./train_eval.sh 1 data/apollo-pifpaf/annotations_loose ${dropout} apolloscape data/apollo-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/apollo-pifpaf/annotations ${dropout} apolloscape data/apollo-pifpaf/annotations "${args}"


export process_mode='mean'

    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations_loose ${dropout} kitti data/kitti-pifpaf/annotations_loose "${args}"
    args='--confidence --lstm'

    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"

    args='--confidence --transformer'
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations_loose ${dropout} kitti data/kitti-pifpaf/annotations "${args}"

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

    args='--transformer'

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

    #source ./train_eval.sh 1 data/apollo-pifpaf/annotations ${dropout} apolloscape data/apollo-pifpaf/annotations "${args}"
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations_loose ${dropout} kitti data/kitti-pifpaf/annotations_loose "${args}"
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations_loose ${dropout} kitti data/kitti-pifpaf/annotations_loose "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car_loose ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car_loose ${dropout} kitti data/kitti-pifpaf/annotations_car_loose "${args}"

    args='--transformer --kps_3d'
    #source ./train_eval.sh 1 data/apollo-pifpaf/annotations_loose ${dropout} apolloscape data/apollo-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/apollo-pifpaf/annotations ${dropout} apolloscape data/apollo-pifpaf/annotations "${args}"

    args='--confidence --transformer --scene_refine'

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"
    
    args='--transformer --scene_refine'

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

    args='--transformer --scene_disp'

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations_loose ${dropout} kitti data/kitti-pifpaf/annotations_loose "${args}"

    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

export process_mode='neg'
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations_loose ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"


export process_mode='mean'
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"


export process_mode='NULL_ISNT_IT'
    #source ./train_eval.sh 0 data/kitti-pifpaf/annotations ${dropout} kitti data/kitti-pifpaf/annotations "${args}"
    #source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car ${dropout} kitti data/kitti-pifpaf/annotations_car "${args}"

#source ./train_eval.sh 1 data/kitti-pifpaf/annotations_car_right ${dropout} kitti data/kitti-pifpaf/annotations_car_right "${args}"


