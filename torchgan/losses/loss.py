import torch
import torch.nn as nn
import copy
from torch import autograd
from math import log
__all__ = ["GeneratorLoss", "DiscriminatorLoss","ProximalDiscriminatorLoss","ProximalGeneratorLoss"]


class GeneratorLoss(nn.Module):
    r"""Base class for all generator losses.

    .. note:: All Losses meant to be minimized for optimizing the Generator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction="mean", override_train_ops=None,eval_only=False):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.arg_map = {}
        self.eval_only = eval_only

    def set_arg_map(self, value):
        r"""Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_generator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_generator,
        device,
        batch_size,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``generator`` defined in this train_ops
        is as follows:

        1. :math:`fake  = generator(noise)`
        2. :math:`value = discriminator(fake)`
        3. :math:`loss  = loss\_function(value)`
        4. Backpropagate by computing :math:`\nabla loss`
        5. Run a step of the optimizer for generator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(
                generator,
                discriminator,
                optimizer_generator,
                device,
                batch_size,
                labels,
            )
        else:
            if labels is None and generator.label_type == "required":
                raise Exception("GAN model requires labels for training")
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            optimizer_generator.zero_grad()
            if generator.label_type == "generated":
                label_gen = torch.randint(
                    0, generator.num_classes, (batch_size,), device=device
                )
            if generator.label_type == "none":
                fake = generator(noise)
            elif generator.label_type == "required":
                fake = generator(noise, labels)
            elif generator.label_type == "generated":
                fake = generator(noise, label_gen)
            if discriminator.label_type == "none":
                dgz = discriminator(fake)
            else:
                if generator.label_type == "generated":
                    dgz = discriminator(fake, label_gen)
                else:
                    dgz = discriminator(fake, labels)
            loss = self.forward(dgz)
            self.loss    = loss
            
            # print("G Loss : ",loss.item())
            if(not self.eval_only):
                loss.backward()
                optimizer_generator.step()
            # NOTE(avik-pal): This will error if reduction is is 'none'
            return loss.item()


class DiscriminatorLoss(nn.Module):
    r"""Base class for all discriminator losses.

    .. note:: All Losses meant to be minimized for optimizing the Discriminator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction="mean", override_train_ops=None,eval_only=False):
        super(DiscriminatorLoss, self).__init__()
        self.reduction          = reduction
        self.override_train_ops = override_train_ops
        self.arg_map            = {}
        self.eval_only          = eval_only

    def set_arg_map(self, value):
        r"""Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_discriminator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value_1 = discriminator(fake)`
        3. :math:`value_2 = discriminator(real)`
        4. :math:`loss = loss\_function(value_1, value_2)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(
                self,
                generator,
                discriminator,
                optimizer_discriminator,
                real_inputs,
                device,
                labels,
            )
        else:
            if labels is None and (
                generator.label_type == "required"
                or discriminator.label_type == "required"
            ):
                raise Exception("GAN model requires labels for training")
            batch_size = real_inputs.size(0)
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            if generator.label_type == "generated":
                label_gen = torch.randint(
                    0, generator.num_classes, (batch_size,), device=device
                )
            optimizer_discriminator.zero_grad()
            if discriminator.label_type == "none":
                dx = discriminator(real_inputs)
            elif discriminator.label_type == "required":
                dx = discriminator(real_inputs, labels)
            else:
                dx = discriminator(real_inputs, label_gen)
            if generator.label_type == "none":
                fake = generator(noise)
            elif generator.label_type == "required":
                fake = generator(noise, labels)
            else:
                fake = generator(noise, label_gen)
            if discriminator.label_type == "none":
                dgz = discriminator(fake.detach())
            else:
                if generator.label_type == "generated":
                    dgz = discriminator(fake.detach(), label_gen)
                else:
                    dgz = discriminator(fake.detach(), labels)
            loss = self.forward(dx, dgz)
            self.loss    = loss
                
            if(not self.eval_only):
                loss.backward()
                optimizer_discriminator.step()
            # NOTE(avik-pal): This will error if reduction is is 'none'
            return loss.item()


class ProximalDiscriminatorLoss(DiscriminatorLoss):
    r"""Base class for all discriminator losses.

    .. note:: All Losses meant to be minimized for optimizing the Discriminator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction="mean",override_train_ops=None,eval_only=False,lamda_prox=0.10,steps=10):
        super(ProximalDiscriminatorLoss, self).__init__(reduction,override_train_ops,eval_only)
        self.reduction          = reduction
        self.override_train_ops = override_train_ops
        self.arg_map            = {}
        self.eval_only          = eval_only
        self.lamda_prox         = lamda_prox
        self.steps              = steps

    def set_arg_map(self, value):
        r"""Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_discriminator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(
        self,
        generator,
        discriminator,
        optimizer_discriminator,
        real_inputs,
        device,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value_1 = discriminator(fake)`
        3. :math:`value_2 = discriminator(real)`
        4. :math:`loss = loss\_function(value_1, value_2)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """

        if self.override_train_ops is not None:
            return self.override_train_ops(
                self,
                generator,
                discriminator,
                optimizer_discriminator,
                real_inputs,
                device,
                labels,
            )
        else:
            real_inputs.requires_grad = True
            if labels is None and (
                generator.label_type == "required"
                or discriminator.label_type == "required"
            ):
                raise Exception("GAN model requires labels for training")
            
            proximal_discriminator = type(discriminator)(discriminator.in_size,discriminator.in_channels,discriminator.step_channels,discriminator.batchnorm,discriminator.nonlinearity,discriminator.last_nonlinearity,discriminator.label_type).to(device)
            proximal_discriminator.load_state_dict(discriminator.state_dict())

            # for step in range(self.steps):
            batch_size = real_inputs.size(0)

            for step in range(self.steps):
                if self.clip is not None:
                    for p in discriminator.parameters():
                        p.data.clamp_(self.clip[0], self.clip[1])
            
                noise      = torch.randn(batch_size, generator.encoding_dims, device=device)
                
                if generator.label_type == "generated":
                    label_gen = torch.randint(
                        0, generator.num_classes, (batch_size,), device=device
                    )
                
                optimizer_discriminator.zero_grad()
                if discriminator.label_type == "none":
                    dx      = discriminator(real_inputs)
                    dx_prox = proximal_discriminator(real_inputs) 
                elif discriminator.label_type == "required":
                    dx      = discriminator(real_inputs, labels)
                    dx_prox = proximal_discriminator(real_inputs, labels)
                else:
                    dx      = discriminator(real_inputs, label_gen)
                    dx_prox = proximal_discriminator(real_inputs, label_gen)
                
                if generator.label_type == "none":
                    fake = generator(noise)
                
                elif generator.label_type == "required":
                    fake = generator(noise, labels)
                else:
                    fake = generator(noise, label_gen)
                
                if discriminator.label_type == "none":
                    dgz = discriminator(fake.detach())
                
                else:
                    if generator.label_type == "generated":
                        dgz = discriminator(fake.detach(), label_gen)
                    else:
                        dgz = discriminator(fake.detach(), labels)
                
                grad_dx      = autograd.grad(torch.unbind(dx), real_inputs, create_graph=False,retain_graph=True)[0]
                grad_dx_prox = autograd.grad(torch.unbind(dx_prox), real_inputs, create_graph=False,retain_graph=True)[0]
                    
                penalty      = torch.mean(torch.square(torch.norm(grad_dx_prox-grad_dx,dim=(2,3))))
                loss         = self.forward(dx, dgz) + self.lamda_prox*penalty
                self.loss    = loss
                loss.backward()
                optimizer_discriminator.step()
                
            
            return loss.item()



class ProximalGeneratorLoss(GeneratorLoss):
    r"""Base class for all generator losses.

    .. note:: All Losses meant to be minimized for optimizing the Generator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction="mean", override_train_ops=None,eval_only=False,lamda_prox=0.10,steps=10,proximal_discriminator_loss=None):
        super(ProximalGeneratorLoss, self).__init__(reduction,override_train_ops,eval_only)
        self.reduction  = reduction
        self.override_train_ops = override_train_ops
        self.arg_map    = {}
        self.eval_only  = eval_only
        self.lamda_prox = lamda_prox
        self.steps      = steps
        self.proximal_discriminator_loss = proximal_discriminator_loss

    def set_arg_map(self, value):
        r"""Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_generator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(
        self,
        real_inputs,
        generator,
        discriminator,
        optimizer_discriminator,
        optimizer_generator,
        device,
        batch_size,
        labels=None,
    ):
        r"""Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``generator`` defined in this train_ops
        is as follows:

        1. :math:`fake  = generator(noise)`
        2. :math:`value = discriminator(fake)`
        3. :math:`loss  = loss\_function(value)`
        4. Backpropagate by computing :math:`\nabla loss`
        5. Run a step of the optimizer for generator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(
                generator,
                discriminator,
                optimizer_generator,
                device,
                batch_size,
                labels,
            )
        else:

            real_inputs.requires_grad = True
            if labels is None and (
                generator.label_type == "required"
                or discriminator.label_type == "required"
            ):
                raise Exception("GAN model requires labels for training")
            
            prox_discriminator = type(discriminator)(discriminator.in_size,discriminator.in_channels,discriminator.step_channels,discriminator.batchnorm,discriminator.nonlinearity,discriminator.last_nonlinearity,discriminator.label_type).to(device)
            prox_discriminator.load_state_dict(discriminator.state_dict())

            for p,q in zip(discriminator.parameters(),prox_discriminator.parameters()):
                q.requires_grad = p.requires_grad
            
            for step in range(self.steps):
                optimizer_discriminator.zero_grad()

                if hasattr(self.proximal_discriminator_loss,"clip"):
                    for p in discriminator.parameters():
                        p.data.clamp_(self.proximal_discriminator_loss.clip[0], self.proximal_discriminator_loss.clip[1])
            
                batch_size = real_inputs.size(0)
                noise = torch.randn(batch_size, generator.encoding_dims, device=device)
                
                if generator.label_type == "generated":
                    label_gen = torch.randint(
                        0, generator.num_classes, (batch_size,), device=device
                    )
            
                
                if discriminator.label_type == "none":
                    dx      = discriminator(real_inputs)
                    dx_prox = prox_discriminator(real_inputs) 
                elif discriminator.label_type == "required":
                    dx      = discriminator(real_inputs, labels)
                    dx_prox = prox_discriminator(real_inputs, labels)
                else:
                    dx      = discriminator(real_inputs, label_gen)
                    dx_prox = prox_discriminator(real_inputs, label_gen)
                
                if generator.label_type == "none":
                    fake = generator(noise)
                
                elif generator.label_type == "required":
                    fake = generator(noise, labels)
                else:
                    fake = generator(noise, label_gen)
                
                if discriminator.label_type == "none":
                    dgz = discriminator(fake.detach())
                
                else:
                    if generator.label_type == "generated":
                        dgz = discriminator(fake.detach(), label_gen)
                    else:
                        dgz = discriminator(fake.detach(), labels)
                
                grad_dx      = autograd.grad(torch.unbind(dx), real_inputs, create_graph=False,retain_graph=True)[0]
                grad_dx_prox = autograd.grad(torch.unbind(dx_prox), real_inputs, create_graph=False,retain_graph=True)[0]
                
                penalty      = torch.mean(torch.square(torch.norm(grad_dx - grad_dx_prox,dim=(2,3))))
                loss         = self.proximal_discriminator_loss.forward(dx, dgz) + self.lamda_prox*penalty
                
                loss.backward()    
                optimizer_discriminator.step()

            
            if labels is None and generator.label_type == "required":
                raise Exception("GAN model requires labels for training")
            noise = torch.randn(batch_size, generator.encoding_dims, device=device)
            optimizer_generator.zero_grad()
            
            if generator.label_type == "generated":
                label_gen = torch.randint(
                    0, generator.num_classes, (batch_size,), device=device
                )
            if generator.label_type == "none":
                fake = generator(noise)
            elif generator.label_type == "required":
                fake = generator(noise, labels)
            elif generator.label_type == "generated":
                fake = generator(noise, label_gen)
            if discriminator.label_type == "none":
                dgz = discriminator(fake)
            else:
                if generator.label_type == "generated":
                    dgz = discriminator(fake, label_gen)
                else:
                    dgz = discriminator(fake, labels)

            loss = self.forward(dgz)
            self.loss    = loss
                
            if(not self.eval_only):
                loss.backward()
                optimizer_generator.step()
            
            discriminator.load_state_dict(prox_discriminator.state_dict())
            
            # NOTE(avik-pal): This will error if reduction is is 'none'
            return loss.item()

