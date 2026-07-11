#!/bin/bash
# CUDA
export CUDA_VISIBLE_DEVICES=$1
# dataset
DATA_DIR=${DATASET_ROOT:-./med-datasets}
# Suppress tokenizer warnings
export TOKENIZERS_PARALLELISM=false
# Datasets to use
datasets=("kather" "pannuke" "digestpath")
seeds=(1)
SHOTS=8
# model
BACKBONE=vit_b32_plip
# trainer
TRAINER=ZeroshotCLIP
# keywords for evaluation
KEYWORDS=('accuracy' 'confidence' 'ece' 'ace' 'mce' 'ece_kde')

# Build trainer config with specific parameters
BATCH_SIZE=100  # Test batch size
TRAINER_CFG="${BACKBONE}_batch${BATCH_SIZE}"

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Evaluating PLIP on dataset: ${dataset} (seed: ${seed})"
        # evaluates on all classes
        bash scripts/classification/all_zeroshot_plip.sh ${TRAINER} ${TRAINER_CFG} ${dataset} ${DATA_DIR} ${SHOTS} ${seed}
    done
    
    # parse results
    echo "Parsing results for dataset: ${dataset}"
    RESULTS_DIR="output/all_zeroshot/${dataset}/shots_${SHOTS}/${TRAINER}/${TRAINER_CFG}"
    for keyword in "${KEYWORDS[@]}"; do
        python parse_test_res.py ${RESULTS_DIR} --test-log --keyword ${keyword}
    done
done