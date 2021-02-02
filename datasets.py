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


def get_dataloader(name):
    dataloaders = {"MNIST":mnist_dataloader,"CIFAR-10":cifar_dataloader,"CELEB-A":celeba_dataloader}
    return dataloaders[name]

def mnist_dataloader(batch_size=512,img_size=32):
    dataset = dsets.MNIST(
        root="~/torchgan/data/mnist",
        train=True,
        transform=transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        ),
        download=True,
    )
    dataset_size = len(dataset)
    
    train_size   = int(0.90*(dataset_size))
    val_size     = int(0.05*(dataset_size))
    test_size    = int(0.05*(dataset_size))

    train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader   = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    test_loader  = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=16)

    return train_loader, val_loader, test_loader




def celeba_dataloader(batch_size=128,img_size=64):

    dataset =  dsets.CelebA(root='~/torchgan/data/celeba',
                            transform=transforms.Compose([
                                        transforms.CenterCrop(160),
                                        transforms.Resize((img_size, img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]),
                            download=True)
    
    dataset_size = len(dataset)
    val_size     = int(0.01*(dataset_size))
    test_size    = int(0.01*(dataset_size))
    train_size   = dataset_size -(val_size+test_size)
    train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader   = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    test_loader  = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=16)

    return train_loader, val_loader, test_loader


def cifar_dataloader(batch_size=512,img_size=32):

    dataset =  dsets.CIFAR10(
                                  root='~/torchgan/data/cifar10',
                                  transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize((img_size, img_size)),
                                        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))]),
                                  download=True)
    
    dataset_size = len(dataset)
    
    dataset_size = len(dataset)
    val_size     = int(0.05*(dataset_size))
    test_size    = int(0.05*(dataset_size))
    train_size   = dataset_size -(val_size+test_size)
    
    train_dataset,val_dataset,test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_loader   = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    test_loader  = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=16)

    return train_loader, val_loader, test_loader
