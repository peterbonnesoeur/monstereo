#!/bin/bash -l


#? Script made to automate the preprocessing, training and evaluation of the model


use_car="$1"

#? Launch the hyperparameter optimization (and set its multiplier)
hyp='0'
multipler='10'

eval_mode='--save --verbose'

#? usage of stereo model
stereo='0'

epochs='80'

#? width of the network
hidden_size='1024' 

#lr='0.01'
lr='0.001'
#lr='0.0001'
#lr='0.001'

#? If the joints are already precomputed, set joints_there to 1 to remove the preprocessing step
joints_there='1'

dropout="$3"

joints_stereo='data/arrays/joints-kitti-stereo-transformer-210123-183442.json'
joints_mono='data/arrays/joints-kitti-transformer-210110-091301.json'

dir_ann="$2"

joints_stereo_car='data/arrays/joints-kitti-vehicles-stereo-201022-1536.json'


joints_mono_car='data/arrays/joints-kitti-vehicles-transformer-210110-092355.json' 

dir_ann="$2"

dataset="$4"

dir_ann_eval="$5"

args="$6"

echo "Dataset:  ${dataset}"
if [ $dataset == "apolloscape" ]
then 
    echo "Apolloscape joints"
    joints_mono_car='data/arrays/joints-apolloscape-train-transformer-210210-103321.json'
fi

if [ $dataset == "nuscenes" ]
then 
    echo "Nuscenes joints"
    joints_mono_car=''
fi


if [$dropout == '']; then
    dropout='0'
fi

model_out () {
    #? Get the name of the model at the output of the training step
    while read -r line; do
        id=`echo $line | cut -c 1-12`   
        echo "$line" 

        if [ $id == "data/models/" ]    
        then     
        model=$line  
        fi 

        if [ $hyp == "1" ]
        then
        hidden_size=$line
        echo "$hidden_size"
        fi
    done <<< "$1"
}


joints_out () {
    #? Get the name of the joints at the output of the prep step
    while read -r line; do
        id=`echo -e $line | cut -c 1-18`   
        echo "$line"
        if [ $id == "data/arrays/joints" ]
        then
        joints=$line  
        fi
    done <<< "$1"
}


if [ $use_car == "1" ]
    then 
    #? the network is working on the vehicle instances
    echo "CAR MODE"

    if [ $joints_there == "0" ]
    then

        echo "Command joints mono processing"
        echo "python3 -m  monstereo.run prep --dir_ann ${dir_ann} --monocular --vehicles --dataset ${dataset} --dropout ${dropout} ${args}"

        output=$(python3 -m  monstereo.run prep --dir_ann ${dir_ann} --monocular --vehicles --dataset ${dataset} --dropout ${dropout} ${args})
        joints_out "$output"
        joints_mono_car="$joints"
    fi
    echo "Output joint file mono"
    echo "$joints_mono_car"



    echo "Command training mono"
    if [ $hyp == "1" ]
    then 
        echo "Hyper pyrameter optimization enabled with multiplier of ${multipler}"
        echo "python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args} --dir_ann ${dir_ann_eval}"
        output=$(python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args} --dir_ann ${dir_ann_eval})
    else
        echo "python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save ${args}"
        output=$(python3 -m  monstereo.run train  --epochs ${epochs} --lr ${lr} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save ${args})
    fi
    model_out "$output"
    model_mono="$model"
    echo "Output mono car model"
    echo "$model_mono"
    echo "$hidden_size"




    if [ $stereo == "1" ]
    then 

        if [ $joints_there == "0" ]
        then

            echo "Command joints stereo processing"
            echo "python3 -m  monstereo.run prep --dir_ann ${dir_ann} --vehicles --dataset ${dataset} --dropout $dropout ${args}"

            output=$(python3 -m  monstereo.run prep --dir_ann ${dir_ann} --vehicles --dataset ${dataset} --dropout $dropout ${args})
            joints_out "$output"
            joints_stereo_car="$joints"
        fi
        echo "Output joint file stereo"
        echo "$joints_mono_car"



        echo "Command training stereo"
        if [ $hyp == "1" ]
        then
            echo "Hyper pyrameter optimization enabled with multiplier of ${multipler}"
            echo "python3 -m  monstereo.run train --epochs ${epochs}  --lr ${lr} --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args}"
            output=$(python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args})
        else
            echo "python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save ${args}"
            output=$(python3 -m  monstereo.run train --epochs ${epochs}  --lr ${lr} --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save ${args})
        fi

        model_out "$output"
        model_stereo="$model"

        echo "output stereo car model"
        echo "$model_stereo"
    else
        model_stereo="nope"
    fi

    if [ $dataset == "kitti" ]
    then
        echo "Generate and evaluate the output"
        echo "python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo} --model_mono ${model_mono} --hidden_size ${hidden_size} --vehicles --generate ${eval_mode} ${args}"
        python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo} --model_mono ${model_mono} --hidden_size ${hidden_size} --vehicles --generate ${eval_mode} ${args}
    fi

else
    #? the network is working on the pedestrian instances
    echo "HUMAN MODE"
    
    if [ $stereo == "0" ]
    then 
        if [ $joints_there == "0" ]
        then
            echo "Command joints mono processing"
            echo "python3 -m  monstereo.run prep --dir_ann ${dir_ann} --monocular --dataset ${dataset} --dropout $dropout ${args}"
            output=$(python3 -m  monstereo.run prep --dir_ann ${dir_ann} --monocular --dataset ${dataset} --dropout $dropout ${args})
            joints_out "$output"
            joints_mono="$joints"
        fi
        echo "Output joint file mono"
        echo "$joints_mono"


        echo "Train mono "
        if [ $hyp == "1" ]
        then
            echo "python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --dataset ${dataset} --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti --hyp --multiplier ${multipler} --save ${args} --dir_ann ${dir_ann_eval}"
            output=$(python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --dataset ${dataset} --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti  --hyp --multiplier ${multipler} --save ${args} --dir_ann ${dir_ann_eval})
        else
            echo " python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --dataset ${dataset} --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti --save ${args}"
            output=$(python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --dataset ${dataset} --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti --save ${args})
        fi
        # train mono model
        
        model_out "$output"
        model_mono="$model"
        echo "$model_mono"
    else
        model_mono="nope"
    fi


    if [ $stereo == "1" ]
    then 

        if [ $joints_there == "0" ]
        then
            echo "Command joints stereo processing"
            echo "python3 -m  monstereo.run prep --dir_ann ${dir_ann} --dataset ${dataset} --dropout $dropout ${args}"
            output=$(python3 -m  monstereo.run prep --dir_ann ${dir_ann} --dataset ${dataset} --dropout $dropout ${args})
            joints_out "$output"
            joints_stereo="$joints"
        fi
        echo "Output joint file mono"
        echo "$joints_stereo"


        echo "Train stereo"
        echo "python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --dataset ${dataset} --joints ${joints_stereo} --hidden_size ${hidden_size} --dataset kitti --save ${args}"
        output=$(python3 -m  monstereo.run train --epochs ${epochs} --lr ${lr} --dataset ${dataset} --joints ${joints_stereo} --hidden_size ${hidden_size} --dataset kitti --save ${args})
        model_out "$output"
        model_stereo="$model"
    else
        model_stereo="nope"
    fi

    echo "python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo}  --model_mono ${model_mono} --generate ${eval_mode} ${args}"
    python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo}  --model_mono ${model_mono} --generate ${eval_mode} ${args}

fi