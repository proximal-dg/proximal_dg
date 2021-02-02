import os
import sys
import torch
from torchgan import *
from torchgan.losses import *
from torchgan.metrics import *
from torchgan.models import *
from torchgan.trainer import *
from datasets import *
from config   import *
import pickle
import argparse
import json

torch.manual_seed(0)

def run(exp_dir,network_params,dataloader,losses_list,img_size=32,epochs=100,ncritic=1,sample_size=100,nrow=10,batch_size=64,device=0,resume=False): 
    
    log_dir     = os.path.join(exp_dir,"logs")
    recon_dir   = os.path.join(exp_dir,"images")
    ckpt_dir    = os.path.join(exp_dir,"checkpoints")
    metric_dir  = os.path.join(exp_dir,"metrics")

    exp_dirs = [log_dir,recon_dir,ckpt_dir,metric_dir]
    for dirname in exp_dirs:
        os.makedirs(dirname,exist_ok=True)

    with open(os.path.join(exp_dir,"network_params"),"wb") as f:
        pickle.dump(network_params,f)
    f.close()
    
    with open(os.path.join(exp_dir,"losses_list"),"wb") as f:
        pickle.dump(losses_list,f)
    f.close()
    
    train_loader, val_loader, test_loader = dataloader(batch_size=batch_size,img_size=img_size)
    metrics_list= [ClassifierScore()]#,FrechetDistance(dataloader=train_loader)]

    if("proximal" in exp_dir):
        trainer = ProximalTrainer(
            network_params,
            losses_list,
            device          =torch.device("cuda:{}".format(device)),
            metrics_list    =metrics_list,
            checkpoints     =ckpt_dir+"/iter_",
            recon           =recon_dir,
            log_dir         =log_dir,
            sample_size     =sample_size,
            nrow            =nrow,
            epochs          =epochs,
            ncritic         =ncritic
        )

    else:
        trainer = Trainer(
            network_params,
            losses_list,
            device          =torch.device("cuda:{}".format(device)),
            metrics_list    =metrics_list,
            checkpoints     =ckpt_dir+"/iter_",
            recon           =recon_dir,
            log_dir         =log_dir,
            sample_size     =sample_size,
            nrow            =nrow,
            epochs          =epochs,
            ncritic         =ncritic
        )
    
    if(resume and os.path.exists(ckpt_dir+"/iter_{}.model".format(epochs-1))):
        trainer.start_epoch = epochs
        trainer.epochs      += epochs 
        trainer.load_model(ckpt_dir+"/iter_{}.model".format(epochs-1))
    trainer(train_loader)

    for name, metric in trainer.metrics.items():
        metric_logs = trainer.logger.get_metric_viz()        
        with open(os.path.join(metric_dir,name),"wb") as f:
            pickle.dump(metric_logs.logs[name],f)

if __name__  == "__main__":

    parser          = argparse.ArgumentParser()
    parser.add_argument('--arg', type=json.loads)
    exp_args        = parser.parse_args().arg
    exp_name        = "divergence"      if "exp_name" not in exp_args else exp_args["exp_name"]
    model           = "SNGAN"    if "model" not in exp_args else exp_args["model"]
    dataset         = "MNIST"    if "dataset" not in exp_args else exp_args["dataset"]
    loss_type       = "minimax"  if "loss_type" not in exp_args else exp_args["loss_type"]
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
    lamda_prox      = 0.10       if "lamda_prox" not in exp_args else exp_args["lamda_prox"]

    network_params  = get_network_param(name=model,img_size=img_size,step_channels=step_channels,out_channels=channels,in_channels=channels)
    losses_list     = get_losses_list(loss_type,lamda_prox)
    dataloader      = get_dataloader(dataset)
    exp_dir         = os.path.join("exp",model,dataset,exp_name)

    if("proximal" in exp_name):
        exp_dir = os.path.join(exp_dir,"lamda_{}".format(lamda_prox))
    run(exp_dir,network_params,dataloader,losses_list,img_size,epochs,ncritic,sample_size,nrow,batch_size,gpu,resume)