import torch
import torch.nn.functional as F
import torchvision

from ..utils import reduce
from .metric import EvaluationMetric

__all__ = ["ClassifierScore"]


class ClassifierScore(EvaluationMetric):
    r"""
    Computes the Classifier Score of a Model. Also popularly known as the Inception Score.
    The ``classifier`` can be any model. It also supports models outside of torchvision models.
    For more details on how to use custom trained models look up the tutorials.

    Args:
        classifier (torch.nn.Module, optional) : The model to be used as a base to compute the classifier
            score. If ``None`` is passed the pretrained ``torchvision.models.inception_v3`` is used.

            .. note ::
                Ensure that the classifier is on the same ``device`` as the Trainer to avoid sudden
                crash.
        transform (torchvision.transforms, optional) : Transformations applied to the image before feeding
            it to the classifier. Look up the documentation of the torchvision models for this transforms.
        sample_size (int): Batch Size for calculation of Classifier Score.
    """

    def __init__(self, classifier=None, transform=None, sample_size=128):
        super(ClassifierScore, self).__init__()
        self.classifier = (
            torchvision.models.inception_v3(True) if classifier is None else classifier
        )
        self.classifier.eval()
        self.transform = transform
        self.sample_size = sample_size
        self.set_arg_map({"generator":"generator" , "device":"device"})
        self.history = []
        
    def preprocess(self, x):
        r"""
        Preprocessor for the Classifier Score. It transforms the image as per the transform requirements
        and feeds it to the classifier.

        Args:
            x (torch.Tensor) : Image in tensor format

        Returns:
            The output from the classifier.
        """
        
        # x = F.interpolate(x, size=(299, 299), mode='bilinear',align_corners=False)

        if(x.size()[1]==1):
            gray_to_rgb  = torchvision.transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
            x = gray_to_rgb(x)

        x = x if self.transform is None else self.transform(x)
        # x = x.to(next(self.classifier.parameters()).device)
        return x

    def calculate_score(self, x):
        r"""
        Computes the Inception Score for the Input.

        Args:
            x (torch.Tensor) : Image in tensor format

        Returns:
            The Inception Score.
        """
        # p = F.softmax(x, dim=1)
        # q = torch.mean(p, dim=0)
        # kl = torch.sum(p * (F.log_softmax(x, dim=1) - torch.log(q)), dim=1)
        # return torch.exp(reduce(kl, "mean")).data
        mean,std = inception_score(x)
        return mean

    def metric_ops(self, generator, device):
        r"""Defines the set of operations necessary to compute the ClassifierScore.

        Args:
            generator (torchgan.models.Generator): The generator which needs to be evaluated.
            device (torch.device)                : Device on which the generator is present.

        Returns:
            The Classifier Score (scalar quantity)
        """
        noise = torch.randn(self.sample_size, generator.encoding_dims, device=device)
        img = generator(noise).detach()
        score = self.__call__(img)
        self.history.append(score)
        
        return score


import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = x.to(next(inception_model.parameters()).device)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)