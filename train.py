import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.advzsclip
import trainers.advcoop
import trainers.advvpt
import trainers.advvpt_moe
import trainers.advmaple
import trainers.advmaple_moe
import trainers.adv_independentVL
import trainers.adv_independentVL_moe

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


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # Config for AT
    cfg.AT = CN()
    cfg.AT.TRAIN = CN()
    cfg.AT.TRAIN.EPS = 4
    cfg.AT.TRAIN.ALPHA = 1
    cfg.AT.TRAIN.STEPS = 3
    cfg.AT.TRAIN.AT_LOSS_TYPE = "adv_half"
    cfg.AT.TEST = CN()
    cfg.AT.TEST.EPS = 4
    cfg.AT.TEST.ALPHA = 1
    cfg.AT.TEST.STEPS = 100

    # Config for AdvCoOp
    cfg.TRAINER.AdvCoOp = CN()
    cfg.TRAINER.AdvCoOp.N_CTX = 16  # number of context vectors
    cfg.TRAINER.AdvCoOp.CSC = False  # class-specific context
    cfg.TRAINER.AdvCoOp.CTX_INIT = ""  # initialization words
    cfg.TRAINER.AdvCoOp.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.AdvCoOp.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # Config for Adversarial Visual-Only Prompt (AdvVPT)
    cfg.TRAINER.AdvVPT = CN()
    cfg.TRAINER.AdvVPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.AdvVPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.AdvVPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.AdvVPT.PROMPT_DEPTH_VISION = 9  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for MoE Adversarial Visual-Only Prompt (MoEAdvVPT)
    cfg.TRAINER.MoEAdvVPT = CN()
    cfg.TRAINER.MoEAdvVPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.MoEAdvVPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MoEAdvVPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MoEAdvVPT.PROMPT_DEPTH_VISION = 9  # if set to 1, will represent shallow vision prompting only
    cfg.TRAINER.MoEAdvVPT.NUM_EXPERTS = 3  # number of experts
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for Adversarial V-L Joint Prompt (AdvMaPLe)
    cfg.TRAINER.AdvMaPLe = CN()
    cfg.TRAINER.AdvMaPLe.N_CTX = 2  # number of context vectors
    cfg.TRAINER.AdvMaPLe.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.AdvMaPLe.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.AdvMaPLe.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for MoE Adversarial V-L Joint Prompt (MoEAdvMaPLe)
    cfg.TRAINER.MoEAdvMaPLe = CN()
    cfg.TRAINER.MoEAdvMaPLe.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MoEAdvMaPLe.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MoEAdvMaPLe.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MoEAdvMaPLe.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.MoEAdvMaPLe.NUM_EXPERTS = 3  # number of experts
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for Adversarial V-L Independent Prompt (AdvIVLP)
    cfg.TRAINER.AdvIVLP = CN()
    cfg.TRAINER.AdvIVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.AdvIVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.AdvIVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.AdvIVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.AdvIVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.AdvIVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for MoE Adversarial V-L Independent Prompt (MoEAdvIVLP)
    cfg.TRAINER.MoEAdvIVLP = CN()
    cfg.TRAINER.MoEAdvIVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.MoEAdvIVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.MoEAdvIVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.MoEAdvIVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.MoEAdvIVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.MoEAdvIVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.MoEAdvIVLP.NUM_EXPERTS = 3  # number of experts
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg

def configure_attack(cfg):
    eps   = cfg.AT.TEST.EPS / 255.0
    alpha = cfg.AT.TEST.ALPHA / 255.0
    steps = cfg.AT.TEST.STEPS
    print("Attack: {}, Test_eps: {}, Test_alpha: {}, Test_steps: {}".format(args.attacks, eps, alpha, steps))
    return eps, alpha, steps

def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()
        print('---------------------------------------------------')
        print('robust acc:')
        eps, alpha, steps = configure_attack(cfg)
        trainer.before_adv_test(attack=args.attacks, eps=eps, alpha=alpha, steps=steps)
        trainer.test_adv()
        return

    if not args.no_train:
        trainer.train()
        print('---------------------------------------------------')
        print('clean acc:')
        trainer.test()
        print('---------------------------------------------------')
        print('robust acc:')
        eps, alpha, steps = configure_attack(cfg)
        trainer.before_adv_test(attack=args.attacks, eps=eps, alpha=alpha, steps=steps)
        trainer.test_adv()

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
    parser.add_argument("--attacks", type=str, default="pgd", help="name of adversarial attack")
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
