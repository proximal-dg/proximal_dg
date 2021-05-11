import os
import sys
import torch.nn as nn
import torch
import torchvision
from torchgan import *
from torchgan.losses import *
from torchgan.metrics import *
from torchgan.models import *
from torchgan.trainer import Trainer
import pickle
from datasets import *
from config   import * 
import argparse
import  json

torch.manual_seed(0)

def evaluate(metric_name="duality_gap",exp_dir=None,losses_list=None,dg_losses_list=None,network_params=None,epochs=100,dataset_loader=None,img_size=32,resume=False):

    log_dir     = os.path.join(exp_dir,"logs")
    recon_dir   = os.path.join(exp_dir,"images")
    ckpt_dir    = os.path.join(exp_dir,"checkpoints")
    metric_dir  = os.path.join(exp_dir,"metrics")
    
    eval_metrics= {}
    
    print(dg_losses_list)
    if(dg_losses_list is None):
        dg_losses_list = losses_list

    train_loader, val_loader, test_loader = dataset_loader(img_size=img_size)

    
    eval_metrics["duality_gap"] = DualityGap(network_params=network_params,
                                            train_dataloader    =val_loader,
                                            eval_dataloader     =test_loader,
                                            generator_loss      =dg_losses_list[0],
                                            discriminator_loss  =dg_losses_list[1],
                                            evaluation_loss     =dg_losses_list[1],
                                            log_dir             =exp_dir,
                                            verbose             =True,
                                            n_iter              =10)

    eval_metrics["proximal_duality_gap"] = ProximalDualityGap(network_params     = network_params,
                                                            train_dataloader      = val_loader,
                                                            eval_dataloader       = test_loader,
                                                            generator_loss        = dg_losses_list[0],
                                                            discriminator_loss    = dg_losses_list[1],
                                                            evaluation_loss       = dg_losses_list[1],
                                                            log_dir               = exp_dir,
                                                            verbose               = True,
                                                            n_iter                = 10)

    metrics_list= [eval_metrics[metric_name]]

    trainer = Trainer(
        network_params,
        dg_losses_list,
        device=torch.device("cuda:0"),
        metrics_list=metrics_list,
        checkpoints=ckpt_dir+"/eval_iter_",
        recon=recon_dir,
        log_dir=log_dir,
        sample_size=100,
        nrow=10,
        epochs=epochs,
        ncritic=1
    )

    save_data = {}
    history   = {}

    for name, metric in trainer.metrics.items():
        save_data[name] = []

    for epoch in range(2,epochs+1):
        for name, metric in trainer.metrics.items():
                metric_logs = trainer.logger.get_metric_viz()
                score =  metric.metric_ops(ckpt_dir+"/iter_",epoch)
                metric_logs.logs[name].append(score)
                save_data[name].append(score)


if __name__  == "__main__":
    
    parser          = argparse.ArgumentParser()
    parser.add_argument('--arg', type=json.loads)
    exp_args        = parser.parse_args().arg
    metric_name     = "proximal_duality_gap"      if "metric_name" not in exp_args else exp_args["metric_name"]
    exp_name        = "convergence"      if "exp_name" not in exp_args else exp_args["exp_name"]
    model           = "SNGAN"  if "model" not in exp_args else exp_args["model"]
    dataset         = "CIFAR-10" if "dataset" not in exp_args else exp_args["dataset"]
    loss_type       = "proximal_minimax"  if "loss_type" not in exp_args else exp_args["loss_type"]
    channels        = 3          if "channels" not in exp_args else exp_args["channels"]
    img_size        = 32         if "img_size" not in exp_args else exp_args["img_size"]
    step_channels   = 64         if "step_channels" not in exp_args else exp_args["step_channels"]
    epochs          = 200        if "epochs" not in exp_args else exp_args["epochs"]
    ncritic         = 1          if "ncritic" not in exp_args else exp_args["ncritic"]
    sample_size     = 100        if "sample_size" not in exp_args else exp_args["sample_size"]
    nrow            = 10         if "nrow" not in exp_args else exp_args["nrow"]
    batch_size      = 128        if "batch_size" not in exp_args else exp_args["batch_size"]
    gpu             = 0          if "gpu" not in exp_args else exp_args["gpu"]
    resume          = 0          if "resume" not in exp_args else exp_args["resume"]
    lamda_prox      = 0.01       if "lamda_prox" not in exp_args else exp_args["lamda_prox"]
    steps           = 10         if "steps" not in exp_args else exp_args["steps"]

    
    dataloader      = get_dataloader(dataset)
    exp_dir         = os.path.join("exp",model,dataset,exp_name)

    if("proximal" in exp_name):
        exp_dir = os.path.join(exp_dir,"lamda_{}".format(lamda_prox,steps))
    
    with open(os.path.join(exp_dir,"network_params"),"rb") as f:
            network_params = pickle.load(f)

    with open(os.path.join(exp_dir,"losses_list"),"rb") as f:
            losses_list = pickle.load(f)

    dg_losses_list     = get_losses_list(loss_type)
    evaluate(metric_name=metric_name,exp_dir=exp_dir,losses_list=losses_list,dg_losses_list=dg_losses_list,network_params=network_params,epochs=epochs,dataset_loader=dataloader,img_size=img_size,resume=resume)