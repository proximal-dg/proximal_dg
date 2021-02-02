from torchgan import * 
import os
import sys
import unittest
import torch.nn as nn
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim import Adam, RMSprop
from torchgan import *
from torchgan.losses import *
from torchgan.metrics import *
from torchgan.models import *
from torchgan.trainer import Trainer
from torchsummary import summary
import pickle
from torch.utils.data.dataloader import default_collate
from datasets import *



def get_network_param(name="SNGAN",img_size=64,step_channels=64,out_channels=3,in_channels=1):
    
    params = {}
    params["DCGAN"] = {
        "generator": {
            "name": DCGANGenerator,
            "args": {"out_channels": out_channels,"out_size":img_size, "step_channels": step_channels,"nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh()},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
            },
        },
        "discriminator": {
            "name": DCGANDiscriminator,
            "args": {"in_channels": in_channels,"in_size":img_size, "step_channels": step_channels,"nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh()},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0002, "betas": (0.5, 0.999)},
            },
        },
    }
    

    params["WGAN-WC"]  = {
        "generator": {
            "name": DCGANGenerator,
            "args": {"out_channels": out_channels,"out_size":img_size, "step_channels": step_channels,"last_nonlinearity":nn.Tanh()},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0001, "betas": (0.5, 0.999)},
            },
        },
        "discriminator": {
            "name": DCGANDiscriminator,
            "args": {"in_channels": in_channels,"in_size":img_size, "step_channels": step_channels,"last_nonlinearity":nn.Identity()},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0001, "betas": (0.5, 0.999)},
            },
        },
    }
    
    params["WGAN-GP"]  = {
        "generator": {
            "name": DCGANGenerator,
            "args": {"out_channels": out_channels,"out_size":img_size, "step_channels": step_channels,"last_nonlinearity":nn.Tanh()},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0001, "betas": (0.9, 0.999)},
            },
        },
        "discriminator": {
            "name": DCGANDiscriminator,
            "args": {"in_channels": in_channels,"in_size":img_size, "step_channels": step_channels,"last_nonlinearity":nn.Identity()},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0004, "betas": (0.9, 0.999)},
            },
        },
    }
    

    params["SNGAN"]  = {
        "generator": {
            "name": SNGANGenerator,
            "args": {"out_channels": out_channels, "step_channels": step_channels},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0003, "betas": (0., 0.999)}
            },
        },
        "discriminator": {
            "name": SNGANDiscriminator,
            "args": {"in_channels": in_channels, "step_channels": step_channels},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0001, "betas": (0., 0.999)}
            },
        },
    }

    params["SAGAN"]  = {
        "generator": {
            "name": SAGANGenerator,
            "args": {"out_channels": out_channels, "step_channels": step_channels},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0002, "betas": (0., 0.999)}
            },
        },
        "discriminator": {
            "name": SAGANDiscriminator,
            "args": {"in_channels": in_channels, "step_channels": step_channels},
            "optimizer": {
                "name": Adam,
                "args": {"lr": 0.0001, "betas": (0., 0.999)}
            },
        },
    }

    return params[name]

def get_losses_list(name="minimax",lamda_prox=0.01,steps=15):
    losses = {}
    losses["minimax"]      = [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
    losses["ns_minimax"]   = [MinimaxGeneratorLoss(nonsaturating=True), MinimaxDiscriminatorLoss()]
    losses["wgan"]         = [WassersteinGeneratorLoss(),WassersteinDiscriminatorLoss(clip=(-0.01, 0.01))]
    losses["wgan_gp"]      = [WassersteinDiscriminatorLoss(),WassersteinGradientPenalty(),WassersteinGeneratorLoss()]

    # losses["proximal_minimax"]      = [ProximalMinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()]
    # losses["proximal_ns_minimax"]   = [ProximalMinimaxGeneratorLoss(nonsaturating=True), MinimaxDiscriminatorLoss()]
    # losses["proximal_wgan"]         = [ProximalWassersteinGeneratorLoss(),WassersteinDiscriminatorLoss(clip=(-0.01, 0.01))]
    # losses["proximal_wgan_gp"]      = [ProximalWassersteinGeneratorLoss(),WassersteinDiscriminatorLoss(),WassersteinGradientPenalty()]

    # losses["proximal_minimax"]      = [ProximalMinimaxGeneratorLoss(proximal_discriminator_loss=MinimaxDiscriminatorLoss(),lamda_prox=lamda_prox,steps=steps), MinimaxDiscriminatorLoss()]
    # losses["proximal_ns_minimax"]   = [ProximalMinimaxGeneratorLoss(nonsaturating=True,proximal_discriminator_loss=MinimaxDiscriminatorLoss(),lamda_prox=lamda_prox,steps=steps), MinimaxDiscriminatorLoss()]
    # losses["proximal_wgan"]         = [ProximalWassersteinGeneratorLoss(proximal_discriminator_loss=WassersteinDiscriminatorLoss(clip=(-0.01, 0.01)),lamda_prox=lamda_prox,steps=steps),WassersteinDiscriminatorLoss(clip=(-0.01, 0.01))]
    # losses["proximal_wgan_gp"]      = [ProximalWassersteinGeneratorLoss(proximal_discriminator_loss=WassersteinDiscriminatorLoss(),lamda_prox=lamda_prox,steps=steps),WassersteinDiscriminatorLoss(),WassersteinGradientPenalty()]


    losses["proximal_minimax"]      = [ProximalMinimaxGeneratorLoss(proximal_discriminator_loss=MinimaxDiscriminatorLoss(),lamda_prox=lamda_prox,steps=steps), MinimaxDiscriminatorLoss(), ProximalMinimaxDiscriminatorLoss(lamda_prox=lamda_prox,steps=steps)]
    losses["proximal_ns_minimax"]   = [ProximalMinimaxGeneratorLoss(nonsaturating=True,proximal_discriminator_loss=MinimaxDiscriminatorLoss(),lamda_prox=lamda_prox,steps=steps), MinimaxDiscriminatorLoss(), ProximalMinimaxDiscriminatorLoss(lamda_prox=lamda_prox,steps=steps)]
    losses["proximal_wgan"]         = [ProximalWassersteinGeneratorLoss(proximal_discriminator_loss=WassersteinDiscriminatorLoss(clip=(-0.01, 0.01)),lamda_prox=lamda_prox,steps=steps),WassersteinDiscriminatorLoss(clip=(-0.01, 0.01)),ProximalWassersteinDiscriminatorLoss(clip=(-0.01, 0.01),lamda_prox=lamda_prox,steps=steps)]

    # losses["proximal_minimax_train"]      = [ProximalMinimaxDiscriminatorLoss(lamda_prox=lamda_prox),MinimaxGeneratorLoss(nonsaturating=True)]
    
    return losses[name]