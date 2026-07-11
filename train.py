import argparse
import torch
import os
import json

from dassl.utils import set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from tools.logger import setup_logger

# custom datasets
import datasets.oxford_pets
import datasets.kather
import datasets.digestpath
import datasets.pannuke
import datasets.covid
import datasets.rsna18

# prompt-learning trainers (CLIP/PLIP/QuiltNet, MedCLIP, BioMedCLIP) + zero-shot
import trainers.classification.base_learner
import trainers.classification.coop
import trainers.classification.zsclip
import trainers.classification.coop_medclip
import trainers.classification.coop_biomedclip

# evaluation
import evaluators.vl_evaluator

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    # replace base classification evaluator with V-L evaluator
    cfg.TEST.EVALUATOR = 'VLClassification'

def extend_cfg(cfg):
    """
    Add new config variables.
    """
    from yacs.config import CfgNode as CN

    # Add MODEL configs at the very start
    cfg.MODEL = CN()
    cfg.MODEL.NAME = ""      # For model type (clip, plip, quiltnet)
    cfg.MODEL_ROOT = ""      # For model path
    cfg.MODEL.BACKBONE = CN()  # Create BACKBONE as a CN
    cfg.MODEL.BACKBONE.NAME = ""  # For backbone name
    cfg.DATASET.SUBSAMPLE_CLASSES = "all" 

    # Config for CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # CoOp Loss config: registry of task + calibration losses (CalibPrompt uses SMAC + AS)
    cfg.TRAINER.COOP.LOSS = CN()
    cfg.TRAINER.COOP.LOSS.ENABLED_LOSSES = ['CE']  # Default to CE only
    
    # CE Loss config
    cfg.TRAINER.COOP.LOSS.CE = CN()
    cfg.TRAINER.COOP.LOSS.CE.WEIGHT = 1.0
    
    # DCA Loss config
    cfg.TRAINER.COOP.LOSS.DCA = CN()
    cfg.TRAINER.COOP.LOSS.DCA.WEIGHT = 15.0

    # MDCA Loss config
    cfg.TRAINER.COOP.LOSS.MDCA = CN()
    cfg.TRAINER.COOP.LOSS.MDCA.WEIGHT = 1.0

    # Focal Loss config
    cfg.TRAINER.COOP.LOSS.FL = CN()
    cfg.TRAINER.COOP.LOSS.FL.WEIGHT = 1.0
    cfg.TRAINER.COOP.LOSS.FL.GAMMA = 3.0

    # Label Smoothing config
    cfg.TRAINER.COOP.LOSS.LS = CN()
    cfg.TRAINER.COOP.LOSS.LS.WEIGHT = 1.0
    cfg.TRAINER.COOP.LOSS.LS.ALPHA = 0.05 

    # MMCE Loss
    cfg.TRAINER.COOP.LOSS.MMCE = CN()
    cfg.TRAINER.COOP.LOSS.MMCE.WEIGHT = 1.0

    # Angular Seperation LOSS
    cfg.TRAINER.COOP.LOSS.AS = CN()
    cfg.TRAINER.COOP.LOSS.AS.WEIGHT = 1.0

    # Smoothed Accuracy and Confidence Matching Loss (SMAC)
    cfg.TRAINER.COOP.LOSS.SMAC = CN()
    cfg.TRAINER.COOP.LOSS.SMAC.WEIGHT = 1.0
    cfg.TRAINER.COOP.LOSS.SMAC.ALPHA = 0.05

    # ECCV24 Penalty Loss
    cfg.TRAINER.COOP.LOSS.ECCV_PENALTY = CN()
    cfg.TRAINER.COOP.LOSS.ECCV_PENALTY.WEIGHT = 10.0

    # ECCV24 ZeroShot Loss
    cfg.TRAINER.COOP.LOSS.ECCV_ZS = CN()
    cfg.TRAINER.COOP.LOSS.ECCV_ZS.WEIGHT = 1.0

    # Margin-based Logit Suppression Loss (MbLS)
    cfg.TRAINER.COOP.LOSS.MBLS = CN()
    cfg.TRAINER.COOP.LOSS.MBLS.WEIGHT = 1.0
    cfg.TRAINER.COOP.LOSS.MBLS.MARGIN = 10.0
    cfg.TRAINER.COOP.LOSS.MBLS.ALPHA = 0.1

    # LogitNorm Loss
    cfg.TRAINER.COOP.LOSS.LOGITNORM = CN()
    cfg.TRAINER.COOP.LOSS.LOGITNORM.WEIGHT = 1.0
    cfg.TRAINER.COOP.LOSS.LOGITNORM.TEMPERATURE = 1.0

    # MARGIN_MEAN_VAR Loss config
    cfg.TRAINER.COOP.LOSS.MARGIN_MEAN_VAR = CN()
    cfg.TRAINER.COOP.LOSS.MARGIN_MEAN_VAR.WEIGHT = 1.0

    # TEXT_MOMENT_MATCHING Loss config
    cfg.TRAINER.COOP.LOSS.TEXT_MOMENT_MATCHING = CN()
    cfg.TRAINER.COOP.LOSS.TEXT_MOMENT_MATCHING.WEIGHT = 5.0

    # IMPORTANT: Keep metrics config for evaluator
    cfg.CALIBRATION = CN()
    cfg.CALIBRATION.METRICS = CN()
    cfg.CALIBRATION.METRICS.ECE_BINS = 10  # the number of bins for ece calculation

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the tuning method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
        
    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    
    base_dir = cfg.OUTPUT_DIR
    base_name = 'log.txt'

    setup_logger(os.path.join(base_dir, base_name))

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True #True

    trainer = build_trainer(cfg)

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)