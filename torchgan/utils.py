from __future__ import print_function
from pkgutil import iter_modules
import torch
import scipy.sparse.linalg as linalg
import torch
from torch import autograd
import numpy as np

class JacobianVectorProduct(linalg.LinearOperator):
    def __init__(self, grad, params):
        if isinstance(grad, (list, tuple)):
            grad = list(grad)
            for i, g in enumerate(grad):
                grad[i] = g.view(-1)
            self.grad = torch.cat(grad)
        elif isinstance(grad, torch.Tensor):
            self.grad = grad.view(-1)
        self.shape = (self.grad.size(0), self.grad.size(0))
        self.dtype = np.dtype('Float32')
        self.params = params

    def _matvec(self, v):
        v = torch.Tensor(v)
        if self.grad.is_cuda:
            v = v.cuda()
        grad_vector_product = torch.dot(self.grad, v)
        hv = autograd.grad(grad_vector_product, self.params, retain_graph=True, allow_unused=True)
        _hv = []
        for g, p in zip(hv, self.params):
            if g is None:
                g = torch.zeros_like(p)
            _hv.append(g.contiguous().view(-1))
        hv = torch.cat(_hv)
        return hv.cpu()

def test_hessian_eigenvalues():
    SIZE = 4
    params = torch.rand(SIZE, requires_grad=True)
    loss = (params**2).sum()/2
    grad = autograd.grad(loss, params, create_graph=True)[0]
    A = JacobianVectorProduct(grad, params)
    e = linalg.eigsh(A, k=2)
    return e

def test_jacobian_eigenvalues():
    SIZE = 4
    param_1 = torch.rand(SIZE, requires_grad=True)
    param_2 = torch.rand(SIZE, requires_grad=True)
    loss_1 = (param_1*param_2).sum()
    loss_2 = -(param_1*param_2).sum()
    grad_1 = autograd.grad(loss_1, param_1, create_graph=True)[0]
    grad_2 = autograd.grad(loss_2, param_2, create_graph=True)[0]
    grad = torch.cat([grad_1, grad_2])
    params =[param_1, param_2]
    A = JacobianVectorProduct(grad, params)
    e = linalg.eigs(A, k=2)
    return e

def reduce(x, reduction=None):
    r"""Applies reduction on a torch.Tensor.

    Args:
        x (torch.Tensor): The tensor on which reduction is to be applied.
        reduction (str, optional): The reduction to be applied. If ``mean`` the  mean value of the
            Tensor is returned. If ``sum`` the elements of the Tensor will be summed. If none of the
            above then the Tensor is returning without any change.

    Returns:
        As per the above ``reduction`` convention.
    """
    if reduction == "mean":
        return torch.mean(x)
    elif reduction == "sum":
        return torch.sum(x)
    else:
        return x


def getenv_defaults(module_name):
    r"""Determines if a particular package is installed in the system.

    Args:
        module_name (str): The name of the package to be found.

    Returns:
        1 if package is installed else 0
    """
    return int(module_name in (name for loader, name, ispkg in iter_modules()))
