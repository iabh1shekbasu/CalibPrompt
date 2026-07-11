#!/bin/bash
export TOKENIZERS_PARALLELISM=false
# CUDA
export CUDA_VISIBLE_DEVICES=$1
# dataset
DATA_DIR=${DATASET_ROOT:-./med-datasets} #/mnt/sharedata/ssd/common/datasets/
new_class_datasets=("rsna18" "covid")

seeds=(1) #(1 2 3)

SHOTS=8
# model
BACKBONE=vit_b32_medclip # ("rn50" "rn101" "vit_b32" "vit_b16" "vit_l14" "vit_b32_medclip")

# trainer
TRAINERS=('CoOp_MedCLIP')

# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'mce' 'ace') 
for TRAINER in "${TRAINERS[@]}"; do
# build train cfg
if [ "${TRAINER}" == "CoOp_MedCLIP" ]; then
EPOCH=50
BATCH_SIZE=16
N_CTX=16
else
echo "Unknown trainer: ${TRAINER}"
exit 1
fi
# LOADEP=${EPOCH} # use last epoch
TRAINER_CFG=${BACKBONE}_c${N_CTX}_ep${EPOCH}_batch${BATCH_SIZE} # build trainer cfg
# Extract loss information from config file
# Extract loss information from config file and store in variables
CONFIG_FILE=configs/trainers/${TRAINER}/${TRAINER_CFG}.yaml
LOSS_DIR=$(python -c "
import yaml
config = yaml.safe_load(open('${CONFIG_FILE}'))
losses = '_'.join(config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'])
weights = '_'.join(str(config['TRAINER']['COOP']['LOSS'][loss]['WEIGHT']) for loss in config['TRAINER']['COOP']['LOSS']['ENABLED_LOSSES'])
print(f'losses_{losses}_weights_{weights}')" 2>/dev/null)
for dataset in "${new_class_datasets[@]}"; do
for seed in "${seeds[@]}"; do
# trains and evaluates on all classes
bash scripts/classification/all_fewshot_medclip.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed}
done
for keyword in "${KEYWORDS[@]}"; do
# Using complete directory path with loss information
RESULTS_DIR="output/all/${dataset}/shots_${SHOTS}/${TRAINER}/${TRAINER_CFG}/${LOSS_DIR}"
python parse_test_res.py ${RESULTS_DIR} --test-log --keyword ${keyword} # Added --per-class flag
done
done
done