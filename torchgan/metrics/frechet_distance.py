import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from ..utils import reduce
from .metric import EvaluationMetric
import  torch.nn as nn
from scipy import linalg

__all__ = ["FrechetDistance"]


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

class FrechetDistance(EvaluationMetric):
    r"""
    Computes the FrechetDistance of a Model. Also popularly known as the Frechet Inception Score.
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

    def __init__(self, classifier=None,dataloader=None, transform=None, sample_size=32):
        super(FrechetDistance, self).__init__()
        
        self.classifier  = torchvision.models.inception_v3(True)
        self.classifier.dropout = nn.Identity()
        self.classifier.fc = nn.Identity()
        self.classifier.eval()
        
        self.transform   = transform
        self.sample_size = sample_size
        self.dataloader  = torch.utils.data.DataLoader(dataloader.dataset, batch_size=self.sample_size, shuffle=True)
        self.set_arg_map({"generator":"generator" , "device":"device"})
        self.history     = []
        
    def preprocess(self, x):
        r"""
        Preprocessor for the Classifier Score. It transforms the image as per the transform requirements
        and feeds it to the classifier.

        Args:
            x (torch.Tensor) : Image in tensor format

        Returns:
            The output from the classifier.
        """
        x_real, x_gen = x

        if(x_real.size()[1]==1):
            gray_to_rgb  = torchvision.transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1))
            x_real,x_gen = gray_to_rgb(x_real), gray_to_rgb(x_gen)

        x_real = F.interpolate(x_real, size=(299, 299), mode='bilinear',align_corners=False)
        x_real = x_real if self.transform is None else self.transform(x_real)
        x_real = x_real.to(next(self.classifier.parameters()).device)

        x_gen = F.interpolate(x_gen, size=(299, 299), mode='bilinear',align_corners=False)
        x_gen = x_gen if self.transform is None else self.transform(x_gen)
        x_gen = x_gen.to(next(self.classifier.parameters()).device)

        return (self.classifier(x_real),self.classifier(x_gen))

    def calculate_score(self, x):
        r"""
        Computes the Inception Score for the Input.

        Args:
            x (torch.Tensor) : Image in tensor format

        Returns:
            The Frechet Inception Distance.
        """
        real_activation, fake_activation = x

        if(type(real_activation) is torch.Tensor):
            real_activation = real_activation.cpu().detach().numpy()
        
        if(type(fake_activation) is torch.Tensor):
            fake_activation = fake_activation.cpu().detach().numpy()

        mu_real    = np.mean(real_activation, axis=0)
        sigma_real = np.cov(real_activation, rowvar=False)

        mu_fake    = np.mean(fake_activation, axis=0)
        sigma_fake = np.cov(fake_activation, rowvar=False)

        return calculate_frechet_distance(mu_real,sigma_real,mu_fake,sigma_fake)

    def metric_ops(self, generator, device):
        r"""Defines the set of operations necessary to compute the ClassifierScore.

        Args:
            generator (torchgan.models.Generator): The generator which needs to be evaluated.
            device (torch.device)                : Device on which the generator is present.

        Returns:
            The Classifier Score (scalar quantity)
        """
        noise        =  torch.randn(self.sample_size, generator.encoding_dims, device=device)
        fake_batch   =  generator(noise).detach()

        real_data    =  next(iter(self.dataloader))
        
        if type(real_data) is tuple or type(real_data) is list:
            real_batch = real_data[0]
        else:
            real_batch = real_data
        
        if type(real_batch) is torch.Tensor:
                real_batch = real_batch.detach()
            
        score = self.__call__((real_batch,fake_batch))
        self.history.append(score)
        return score
