#!/usr/bin/env bash

# module load CUDA/8.0.61_375.26-GCC-5.4.0-2.26
# module load CUDA/10.0.130
# module load CUDA/10.1.243-GCC-8.3.0
module load CUDA/11.1.1-GCC-10.2.0
# module load GCC/8.2.0-2.31.1
#source ~/anaconda3/bin/activate py3_torch_cuda8
# source ~/anaconda3/bin/activate py3_torch_cuda10
# source ~/anaconda3/bin/activate torch_only
source ~/anaconda3/bin/activate torch18


#module load CUDA/9.2.88-GCC-7.3.0-2.30
#source ~/anaconda3/bin/activate py3_torch_cuda9

model=$1
cur_fold=$2
encoder_type=$3
# look_forward_ratio=$6
topk_fwr=$4
topk_back=$5
threshold_sim=$6
max_epoch=$7
cv_attempts=${8}
encode_type=${9}
use_coref=${10}
gen_logs=${11}
use_np_focus=${12}

source_domain=${13}
target_domain=${14}


#for fold in {0..4}
#do
#   python ~/workspace/cohe1/main.py --cur_fold $fold --essay_prompt_id_train $source_domain --essay_prompt_id_test $target_domain
    
#done
#python main_cv.py --essay_prompt_id_train $source_domain --essay_prompt_id_test $target_domain --target_model $model --cur_fold $cur_fold --encoder_type $encoder_type --cv_attempts $cv_attempts
python main.py --target_model $model --cur_fold $cur_fold --encoder_type $encoder_type --topk_fwr $topk_fwr --topk_back $topk_back --threshold_sim $threshold_sim --max_epoch $max_epoch --cv_attempts $cv_attempts  --encode_type $encode_type --use_coref $use_coref --gen_logs $gen_logs --use_np_focus $use_np_focus --essay_prompt_id_train $source_domain --essay_prompt_id_test $target_domain
