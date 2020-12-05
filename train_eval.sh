#!/bin/bash -l

command_izar=''
#srun --partition gpu  --reservation=vita-2020-11 --qos gpu --gres gpu:1'

use_car="$1"

hyp='0'

multipler='2'

eval_mode='--save --verbose'

stereo='0'

epochs='400'

hidden_size='1024' 

joints_there='0'

dropout="$3"

joints_stereo='data/arrays/joints-kitti-stereo-201021-1518.json'
joints_mono='data/arrays/joints-kitti-201023-1410.json' 
dir_ann="$2"

joints_stereo_car='data/arrays/joints-kitti-vehicles-stereo-201022-1536.json'
joints_mono_car='data/arrays/joints-kitti-vehicles-201028-1054.json' 

dir_ann_car="$2"

dataset="$4"

dir_ann_eval="$5"

args="$6"

#Test apolloscape

#dataset='apolloscape'
#dir_ann_car='data/apollo-pifpaf/annotations'
#joints_mono_car='/home/maximebonnesoeur/monstereo/data/arrays/joints-apolloscape-train-201028-2236.json'

echo "proceess_mode ${process_mode}"


if [$command_izar != '']; then
    echo "GPU USED ${command_izar}"
fi

if [$dropout == '']; then
    dropout='0'
fi

model_out () {
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

    echo "CAR MODE"

    if [ $joints_there == "0" ]
    then

        echo "Command joints mono processing"
        echo "python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --monocular --vehicles --dataset ${dataset} --dropout ${dropout} ${args}"

        output=$(python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --monocular --vehicles --dataset ${dataset} --dropout ${dropout} ${args})
        joints_out "$output"
        joints_mono_car="$joints"
    fi
    echo "Output joint file mono"
    echo "$joints_mono_car"



    echo "Command training mono"
    if [ $hyp == "1" ]
    then 
        echo "Hyper pyrameter optimization enabled with multiplier of ${multipler}"
        echo "${command_izar}  python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args}"
        output=$(${command_izar}  python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args})
    else
        echo "${command_izar}  python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save ${args}"
        output=$(${command_izar}  python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_mono_car} --hidden_size ${hidden_size} --monocular --vehicles --dataset ${dataset} --save ${args})
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
            echo "${command_izar}  python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --vehicles --dataset ${dataset} --dropout $dropout ${args}"

            output=$(${command_izar}  python3 -m  monstereo.run prep --dir_ann ${dir_ann_car} --vehicles --dataset ${dataset} --dropout $dropout ${args})
            joints_out "$output"
            joints_stereo_car="$joints"
        fi
        echo "Output joint file stereo"
        echo "$joints_mono_car"



        echo "Command training stereo"
        if [ $hyp == "1" ]
        then
            echo "Hyper pyrameter optimization enabled with multiplier of ${multipler}"
            echo "${command_izar}  python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args}"
            output=$(${command_izar}  python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save --hyp --multiplier ${multipler} ${args})
        else
            echo "${command_izar} python3 -m  monstereo.run train --epochs ${epochs} --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save ${args}"
            output=$(${command_izar} python3 -m  monstereo.run train --epochs ${epochs}  --joints ${joints_stereo_car} --hidden_size ${hidden_size} --vehicles --dataset ${dataset} --save ${args})
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
        echo "${command_izar}  python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo} --model_mono ${model_mono} --hidden_size ${hidden_size} --vehicles --generate ${eval_mode} ${args}"
        ${command_izar} python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo} --model_mono ${model_mono} --hidden_size ${hidden_size} --vehicles --generate ${eval_mode} ${args}
    fi

else

    echo "HUMAN MODE"


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


    # train mono model
    echo "Train mono "
    echo 
    echo "${command_izar}  python3 -m  monstereo.run train --epochs ${epochs} --dataset ${dataset} --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti --save ${args}"
    output=$(${command_izar}  python3 -m  monstereo.run train --epochs ${epochs} --dataset ${dataset} --joints ${joints_mono} --hidden_size ${hidden_size} --monocular --dataset kitti --save ${args})
    model_out "$output"

    model_mono="$model"
    echo "$model_mono"


    if [ $stereo == "1" ]
    then 

        if [ $joints_there == "0" ]
        then
            echo "Command joints stereo processing"
            echo "${command_izar}  python3 -m  monstereo.run prep --dir_ann ${dir_ann} --dataset ${dataset} --dropout $dropout ${args}"
            output=$(${command_izar}  python3 -m  monstereo.run prep --dir_ann ${dir_ann} --dataset ${dataset} --dropout $dropout ${args})
            joints_out "$output"
            joints_stereo="$joints"
        fi
        echo "Output joint file mono"
        echo "$joints_stereo"


        echo "Train stereo"
        echo "${command_izar}  python3 -m  monstereo.run train --epochs ${epochs} --dataset ${dataset} --joints ${joints_stereo} --hidden_size ${hidden_size} --dataset kitti --save ${args}"
        output=$(${command_izar}  python3 -m  monstereo.run train --epochs ${epochs} --dataset ${dataset} --joints ${joints_stereo} --hidden_size ${hidden_size} --dataset kitti --save ${args})
        model_out "$output"
        model_stereo="$model"
    else
        model_stereo="nope"
    fi

    echo "${command_izar} python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo}  --model_mono ${model_mono} --generate ${eval_mode} ${args}"
    ${command_izar} python3 -m monstereo.run eval --dir_ann ${dir_ann_eval} --model ${model_stereo}  --model_mono ${model_mono} --generate ${eval_mode} ${args}

fi