#!/bin/bash
# custom config
# trainer
TRAINER=$1
CFG=$2
# dataset
DATASET=$3
DATA=$4
SHOTS=$5
SEED=$6
# calibration config (optional)
CALIBRATION_CFG=$7

# Parse the backbone from CFG (format: backbone_batchX)
BACKBONE=$(echo $CFG | cut -d'_' -f1-2)

# If calibration config is provided, use it
if [ -n "$CALIBRATION_CFG" ]; then
    CALIB_ARG="--calibration-config \"${CALIBRATION_CFG}\""
    # Include calibration config in output directory
    CALIB_DIR=$(basename "$CALIBRATION_CFG" .yaml)
    DIR=output/all_zeroshot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/calib_${CALIB_DIR}/seed${SEED}
else
    CALIB_ARG=""
    DIR=output/all_zeroshot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
fi

mkdir -p ${DIR}

echo "Running PLIP zero-shot evaluation on all classes and saving results to ${DIR}"
python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${BACKBONE}.yaml \
${CALIB_ARG} \
--output-dir ${DIR} \
--eval-only \
DATASET.NUM_SHOTS ${SHOTS} \
MODEL.NAME "plip" \
MODEL_ROOT "${MODEL_ROOT:-./models}" \
DATALOADER.TRAIN_X.BATCH_SIZE 16 \
DATALOADER.TEST.BATCH_SIZE 100 \
DATALOADER.NUM_WORKERS 8