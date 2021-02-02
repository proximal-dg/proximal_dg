import torch
import torch.nn.functional as F
import torchvision
import copy
import time
import os
from ..utils import reduce
from .metric import EvaluationMetric
from torchgan.trainer import *
import torch.multiprocessing as mp
import numpy as np
from torch.optim import Adam

__all__ = ["DualityGap"]


class DualityGap(EvaluationMetric):
    r"""
    Computes the DualityGap of a Model.
    
    Args:
        
        optimizer        : The optimizer to be used for DG estimation ('SGD','Adam')
        n_iter           : The no. epochs in M1 and M2 estimation      (int)
        perturb          : Use perturbed DG                           (Boolean)
    """

    def __init__(self,perturbation=False,network_params=None,generator_loss=None,discriminator_loss=None,evaluation_loss=None,train_dataloader=None,eval_dataloader=None,n_iter=10,log_dir="./",sample_size=25,n_row=5,verbose=False):
    
        super(DualityGap, self).__init__()
        self.perturbation       = perturbation
        self.n_iter             = n_iter
        self.network_params     = network_params
        self.generator_loss     = generator_loss
        self.discriminator_loss = discriminator_loss
        self.evaluation_loss    = evaluation_loss
        self.train_dataloader   = train_dataloader
        self.eval_dataloader    = eval_dataloader if eval_dataloader is not None else train_dataloader
        self.log_dir            = log_dir
        self.sample_size        = sample_size
        self.n_row              = n_row
        self.set_arg_map({"ckpt_dir":"checkpoints" , "ckpt_no":"last_retained_checkpoint"})
        self.evaluation_loss.eval_only = True
        self.verbose            = verbose
        self.history            = []
        self.network_params["generator"]["optimizer"]["name"]=Adam
        self.network_params["discriminator"]["optimizer"]["name"]=Adam
        
    def preprocess(self, x):
        r"""
        Preprocessor for the trainer object

        Args:
            x (torch.Tensor) : Instance of class BaseTrainer

        Returns:
            Trainer class after preprocessing
        """
        return x
    
    def attempt_deviation(self,trainer):
        
        trainer(self.train_dataloader)
        trainer.losses[type(self.evaluation_loss).__name__] = self.evaluation_loss
        trainer._store_loss_maps()

        batch_score = []
        for data in self.eval_dataloader:
            if type(data) is tuple or type(data) is list:
                trainer.real_inputs = data[0].to(trainer.device)
                trainer.labels      = data[1].to(trainer.device)
            elif type(data) is torch.Tensor:
                trainer.real_inputs = data.to(trainer.device)
            else:
                trainer.real_inputs = data
            batch_score.append(-1*self.evaluation_loss.train_ops(**trainer._get_arguments(trainer.loss_arg_maps[type(self.evaluation_loss).__name__]))                    )
        return np.mean(batch_score)

    def calculate_score(self,load_path=None,m1_dir=None,m2_dir=None):
        r"""
        Computes the duality gap for a given trainer instance.

        Args:
            load_path (str) :  Path to load the Instance of class BaseTrainer
            m1_dir (str) :  Path to save the logs for estimating M1
            m2_dir (str) :  Path to save the logs for estimating M2

        Returns:
            The Duality Gap.
        """
        disc_trainer         = Trainer(self.network_params,[self.discriminator_loss],log_dir=os.path.join(m1_dir,"logs"),recon=os.path.join(m1_dir,"images"),checkpoints=os.path.join(m1_dir,"ckpts","model_"),n_critic=1,sample_size=self.sample_size,nrow=self.n_row,verbose=self.verbose)        
        disc_trainer.load_model(load_path,model_only=True)
        disc_trainer.epochs  = self.n_iter
        disc_trainer.loss_information["generator_iters"]    = 1
        disc_trainer.tune_report = "DG"

        gen_trainer         = Trainer(self.network_params,[self.generator_loss],log_dir=os.path.join(m2_dir,"logs"),recon=os.path.join(m2_dir,"images"),checkpoints=os.path.join(m2_dir,"ckpts","model_"),n_critic=1,sample_size=self.sample_size,nrow=self.n_row,verbose=self.verbose)        
        gen_trainer.load_model(load_path,model_only=True)
        gen_trainer.epochs  = self.n_iter
        gen_trainer.loss_information["discriminator_iters"] = 1
        gen_trainer.tune_report = "DG"

        if(self.verbose):
            print("__"*10,"\n{:30s}\n".format("Estimating M1"),"__"*10)
        M1 = self.attempt_deviation(disc_trainer)
        if(self.verbose):
            print("M1 : ",M1)
            print("__"*10,"\n{:30s}\n".format("Estimating M2"),"__"*10)
        M2 = self.attempt_deviation(gen_trainer)
        if(self.verbose):
            print("M2 : ",M2)
        disc_trainer.complete()
        gen_trainer.complete()
        return abs(M1 - M2)

    def metric_ops(self,ckpt_dir=None,ckpt_no=None):
        r"""Defines the set of operations necessary to compute the ClassifierScore.

        Args:
            generator (torchgan.models.Generator): The generator which needs to be evaluated.
            device (torch.device): Device on which the generator is present.

        Returns:
            The Classifier Score (scalar quantity)
        """
        if(self.verbose):
            print("=="*60,"\n{:^120s}\n".format("Estimating Duality Gap"),"=="*60)
        load_path  = ckpt_dir + str(ckpt_no-1)+ ".model"
        m1_dir     =  os.path.join(self.log_dir,"duality_gap","M1","iter_{}".format(ckpt_no))
        m2_dir     =  os.path.join(self.log_dir,"duality_gap","M2","iter_{}".format(ckpt_no))

        start_time = time.time()
        score      = self.calculate_score(load_path=load_path,m1_dir=m1_dir,m2_dir=m2_dir)
        time_taken = time.time()-start_time
        
        self.history.append(score)
        if(self.verbose):
            print("__"*60,"\n{:^50s} : {}\n".format("Duality Gap",score),"__"*60)
        return score
