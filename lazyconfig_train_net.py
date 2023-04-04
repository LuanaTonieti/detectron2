#!/usr/bin/env python3.9
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

import detectron2

from detectron2 import model_zoo

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.data.datasets import register_coco_instances
from detectron2.data.samplers import RepeatFactorTrainingSampler
import sys
sys.path.insert(0, './projects/ViTDet/configs/common')
from projects.ViTDet.configs.common.coco_loader_lsj import dataloader
from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from omegaconf import DictConfig, ListConfig, OmegaConf, SCMode
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import SemSegEvaluator
from projects.ViTDet.configs.COCO.mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

logger = logging.getLogger("detectron2")

# config_file = "./projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py"
config_file = "./projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_b_100ep.py"

path_to_save = "result_robot_cascade_datasets_5000"

dataset_train_name = ["train_al_rihla", "train_beau_jeu", "train_conext_19", "train_conext_21", "train_fracas", "train_star_lancer", "train_telstar", "train_telstar_mechta", "train_torso", "train_uniforia" ]
dataset_test_name = [ "test_al_rihla", "test_beau_jeu", "test_conext_19", "test_conext_21", "test_fracas", "test_star_lancer", "test_telstar", "test_telstar_mechta", "test_torso", "test_uniforia" ]
path_dataset = [ "Al_Rihla", "Beau_Jeu", "Conext_19", "Conext_21", "Fracas", "Star_Lancer", "Telstar", "Telstar_Mechta", "Torso", "Uniforia" ]


def do_test(cfg, model):
    model.eval()
    print(cfg.dataloader.evaluator)
    #cfg.model.backbone.net.img_size = 1024
    for dataset in dataset_test_name:
        cfg.dataloader.test.dataset.names = dataset
        metadata_test = MetadataCatalog.get(cfg.dataloader.test.dataset.names) # to get labels from ids
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
    return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)



def main(args):
    cfg = LazyConfig.load(config_file)
    
    default_setup(cfg, args)


    for i in range (len(dataset_train_name)):
        register_coco_instances(dataset_train_name[i], {},"./ball_dataset/" + path_dataset[i] + "/train/_annotations.coco.json", "./ball_dataset/" + path_dataset[i] + "/train")
        register_coco_instances(dataset_test_name[i], {},"./ball_dataset/" + path_dataset[i] + "/test/_annotations.coco.json", "./ball_dataset/" + path_dataset[i] + "/test")
        MetadataCatalog.get(dataset_train_name[i]).set(thing_classes=["ball", "robot"])
        MetadataCatalog.get(dataset_train_name[i]).thing_classes=["ball", "robot"]
        MetadataCatalog.get(dataset_train_name[i]).set(thing_classes=["ball", "robot"])
        MetadataCatalog.get(dataset_train_name[i]).thing_classes=["ball", "robot"]
        dataset_dicts = DatasetCatalog.get(dataset_train_name[i])



    cfg.dataloader.train.dataset.names = tuple(dataset_train_name)

    metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names) # to get labels from ids

    cfg.dataloader.evaluator.output_dir = "eval_" + path_to_save
    #cfg.dataloader.evaluator = COCOEvaluator("ball_test",tasks={"bbox"},distributed = False, output_dir="ball_train")
    
   
    if args.eval_only:
        train.init_checkpoint= "./" + path_to_save + "/model_final.pth"
        print(train)
        cfg.model.roi_heads.num_classes = 2
        cfg.model.backbone.norm = "BN"
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        cfg.model.backbone.norm = "BN"
        # print(cfg.dataloader.train)
        cfg.train.max_iter=5000
        cfg.train.output_dir= "./" + path_to_save
        cfg.model.roi_heads.num_classes = 2
        cfg.optimizer.lr=0.00005 # 0.0005 para a mask_rcnn e 0.00005 para a cascade
        # cfg.dataloader.train.total_batch_size = 2
        cfg.train.device = "cuda"
        
        cfg = LazyConfig.apply_overrides(cfg, args.opts)
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
