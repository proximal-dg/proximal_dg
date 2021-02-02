import torch
import torch.nn.functional as F
import torchvision
import copy
import time
import os
from ..utils import reduce,JacobianVectorProduct
from .metric import EvaluationMetric
from torchgan.trainer import *
import numpy as np
import matplotlib
import torchvision.utils as vutils
from torch import autograd
from matplotlib import gridspec
import scipy.sparse.linalg as linalg
import pickle

matplotlib.use('Agg')

import matplotlib.pyplot as plt
__all__ = ["EigenVal"]


class EigenVal(EvaluationMetric):
    r"""
    Computes the DualityGap of a Model.
    
    Args:
        
        optimizer        : The optimizer to be used for DG estimation ('SGD','Adam')
        n_iter           : The no. steps in M1 and M2 estimation      (int)
        perturb          : Use perturbed DG                           (Boolean)
    """

    def __init__(self,network_params=None,generator_loss=None,discriminator_loss=None,n_eigs=50,dataloader=None,log_dir="./",sample_size=28,n_row=7,imaginary=False,verbose=False):

        super(EigenVal, self).__init__()
        self.network_params     = network_params
        self.generator_loss     = generator_loss
        self.discriminator_loss = discriminator_loss
        self.dataloader         = dataloader
        self.log_dir            = log_dir
        self.sample_size        = sample_size
        self.n_row              = n_row
        self.set_arg_map({"ckpt_dir":"checkpoints" , "ckpt_no":"last_retained_checkpoint"})
        self.generator_loss.eval_only     = True
        self.discriminator_loss.eval_only = True
        self.n_eigs             = n_eigs
        self.verbose            = verbose
        self.imaginary          = imaginary

    def preprocess(self, x):
        r"""
        Preprocessor for the trainer object

        Args:
            x (torch.Tensor) : Instance of class BaseTrainer

        Returns:
            Trainer class after preprocessing
        """
        return x
    
    def compute_eigenvalues(self,trainer):
        batch_score = []
        start_time = time.time()
        grad_gen_epoch = [torch.zeros_like(p) for p in trainer.generator.parameters()]
        grad_dis_epoch = [torch.zeros_like(p) for p in trainer.discriminator.parameters()]
        n_data = 0    
        n_eigs = self.n_eigs
        for i,data in enumerate(self.dataloader):
            if type(data) is tuple or type(data) is list:
                trainer.real_inputs = data[0].to(trainer.device)
                trainer.labels      = data[1].to(trainer.device)
            elif type(data) is torch.Tensor:
                trainer.real_inputs = data.to(trainer.device)
            else:
                trainer.real_inputs = data
            self.discriminator_loss.train_ops(**trainer._get_arguments(trainer.loss_arg_maps[type(self.discriminator_loss).__name__]))
            self.generator_loss.train_ops(**trainer._get_arguments(trainer.loss_arg_maps[type(self.generator_loss).__name__]))
            dis_loss = self.discriminator_loss.loss 
            gen_loss = self.generator_loss.loss 
            grad_gen = autograd.grad(gen_loss, trainer.generator.parameters(), create_graph=True)
            grad_dis = autograd.grad(dis_loss, trainer.discriminator.parameters(), create_graph=True)
            
            for i, g in enumerate(grad_gen):
                grad_gen_epoch[i] += g * len(data)

            for i, g in enumerate(grad_dis):
                grad_dis_epoch[i] += g * len(data)
            n_data += len(data)

        grad_gen_epoch = [g / n_data for g in grad_gen_epoch]
        grad_dis_epoch = [g / n_data for g in grad_dis_epoch]

        t0 = time.time()
        A = JacobianVectorProduct(grad_gen_epoch, list(trainer.generator.parameters()))
        if  self.imaginary:
            gen_eigs = linalg.eigs(A, k=n_eigs, which='LI')[0]
        else:
            gen_eigs = linalg.eigsh(A, k=n_eigs)[0]
        print("Time to compute Eig-values: %.2f" % (time.time() - t0))

        t0 = time.time()
        A = JacobianVectorProduct(grad_dis_epoch, list(trainer.discriminator.parameters()))
        if  self.imaginary:
            dis_eigs = linalg.eigs(A, k=n_eigs, which='LI')[0]
        else:
            dis_eigs = linalg.eigsh(A, k=n_eigs)[0]
        print("Time to compute Eig-values: %.2f" % (time.time() - t0))

        t0 = time.time()
        grad = grad_gen_epoch + grad_dis_epoch
        params = list(trainer.generator.parameters()) + list(trainer.discriminator.parameters())
        A = JacobianVectorProduct(grad, params)
        if  self.imaginary:
            game_eigs = linalg.eigs(A, k=n_eigs, which='LI')[0]
        else:
            game_eigs = linalg.eigs(A, k=n_eigs)[0]
        print("Time to compute Eig-values: %.2f" % (time.time() - t0))

        if self.verbose:
            print(gen_eigs[:5])
            print(dis_eigs[:5])
            print(game_eigs[:5])
            print("Time to finish: %.2f minutes" % ((time.time() - start_time) / 60.))

        return gen_eigs, dis_eigs, game_eigs

    def calculate_score(self,load_path=None,save_dir=None,step=0):
        r"""
        Computes the duality gap for a given trainer instance.

        Args:
            load_path (str) :  Path to load the Instance of class BaseTrainer
            m1_dir (str) :  Path to save the logs for estimating M1
            m2_dir (str) :  Path to save the logs for estimating M2

        Returns:
            The Duality Gap.
        """
        trainer         = Trainer(self.network_params,[self.discriminator_loss,self.generator_loss],log_dir=os.path.join(save_dir,"logs"),recon=os.path.join(save_dir,"images"),checkpoints=os.path.join(save_dir,"ckpts","model_"),sample_size=self.sample_size,nrow=self.n_row)        
        trainer.load_model(load_path,model_only=True)

        print("__"*10,"\n{:30s}\n".format("Computing Eigen Values"),"__"*10)
        gen_eigs, dis_eigs, game_eigs = self.compute_eigenvalues(trainer)
        print("gen_eigs : ",gen_eigs)
        print("dis_eigs : ",dis_eigs)
        print("game_eigs : ",game_eigs)
        self.plot_eigenvalues(gen_eigs, dis_eigs, game_eigs,out_dir=save_dir,summary_writer=trainer.logger.writer,step=step)
        
        return gen_eigs, dis_eigs, game_eigs

    def metric_ops(self,ckpt_dir=None,ckpt_no=None):
        r"""Defines the set of operations necessary to compute the ClassifierScore.

        Args:
            generator (torchgan.models.Generator): The generator which needs to be evaluated.
            device (torch.device): Device on which the generator is present.

        Returns:
            The Eigen Values
        """
        print("=="*60,"\n{:^120s}\n".format("Estimating Eigen Values"),"=="*60)
        load_path  = ckpt_dir + str(ckpt_no-1)+ ".model"
        save_dir   =  os.path.join(self.log_dir,"eigen_val","iter_{}".format(ckpt_no))
        
        start_time = time.time()
        self.gen_eigs, self.dis_eigs, self.game_eigs = self.calculate_score(load_path=load_path,save_dir=save_dir,step=ckpt_no-1)
        time_taken = time.time()-start_time
        return 0

    def plot_eigenvalues(self,gen_eigs, dis_eigs, game_eigs, labels=None, out_dir=None, summary_writer=None, step=0):
        """
        plots interpolation path in `hist` and computed by `compute_path_stats`.
        """
        assert out_dir is not None or summary_writer is not None, 'save results either as files in out_dir or in tensorboard!'

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fig1 = plt.figure()
        for i, eigs in enumerate(game_eigs):
            plt.scatter(eigs.real, eigs.imag)
        plt.legend()

        fig2 = plt.figure()
        plt.bar(np.arange(len(gen_eigs)), gen_eigs[::-1])
        plt.legend()

        fig3 = plt.figure()
        plt.bar(np.arange(len(dis_eigs)), dis_eigs[::-1])
        plt.legend()

        if out_dir is not None:
            fig1.savefig(os.path.join(out_dir, 'game_eigs_%06d.png' % step))
            fig2.savefig(os.path.join(out_dir, 'gen_eigs_%06d.png' % step))
            fig3.savefig(os.path.join(out_dir, 'dis_eigs_%06d.png' % step))

            with open(os.path.join(out_dir,"data"),"wb") as f:
                pickle.dump({"gen_eigs":gen_eigs,"dis_eigs":dis_eigs,"game_eigs":game_eigs},f)
                
        if summary_writer is not None:
            summary_writer.add_figure('game_eigs', fig1, step)
            summary_writer.add_figure('gen_eigs', fig2, step)
            summary_writer.add_figure('dis_eigs', fig3, step)