import torch

from ..logging.logger import Logger
from ..losses.loss import DiscriminatorLoss, GeneratorLoss
from ..models.model import Discriminator, Generator
from .base_trainer import BaseTrainer

__all__ = ["ProximalTrainer"]


class ProximalTrainer(BaseTrainer):
    r"""Standard Trainer for various GANs. This has been designed to work only on one GPU in case
    you are using a GPU.

    Most of the functionalities provided by the Trainer are flexible enough and can be customized by
    simply passing different arguments. You can train anything from a simple DCGAN to complex CycleGANs
    without ever having to subclass this ``Trainer``.

    Args:
        models (dict): A dictionary containing a mapping between the variable name, storing the
            ``generator``, ``discriminator`` and any other model that you might want to define, with the
            function and arguments that are needed to construct the model. Refer to the examples to
            see how to define complex models using this API.
        losses_list (list): A list of the Loss Functions that need to be minimized. For a list of
            pre-defined losses look at :mod:`torchgan.losses`. All losses in the list must be a
            subclass of atleast ``GeneratorLoss`` or ``DiscriminatorLoss``.
        metrics_list (list, optional): List of Metric Functions that need to be logged. For a list of
            pre-defined metrics look at :mod:`torchgan.metrics`. All losses in the list must be a
            subclass of ``EvaluationMetric``.
        device (torch.device, optional): Device in which the operation is to be carried out. If you
            are using a CPU machine make sure that you change it for proper functioning.
        ncritic (int, optional): Setting it to a value will make the discriminator train that many
            times more than the generator. If it is set to a negative value the generator will be
            trained that many times more than the discriminator.
        sample_size (int, optional): Total number of images to be generated at the end of an epoch
            for logging purposes.
        epochs (int, optional): Total number of epochs for which the models are to be trained.
        checkpoints (str, optional): Path where the models are to be saved. The naming convention is
            if checkpoints is ``./model/gan`` then models are saved as ``./model/gan0.model`` and so on.
        retain_checkpoints (int, optional): Total number of checkpoints that should be retained. For
            example, if the value is set to 3, we save at most 3 models and start rewriting the models
            after that.
        recon (str, optional): Directory where the sampled images are saved. Make sure the directory
            exists from beforehand.
        log_dir (str, optional): The directory for logging tensorboard. It is ignored if
            TENSORBOARD_LOGGING is 0.
        test_noise (torch.Tensor, optional): If provided then it will be used as the noise for image
            sampling.
        nrow (int, optional): Number of rows in which the image is to be stored.

    Any other argument that you need to store in the object can be simply passed via keyword arguments.

    Example:
        >>> dcgan = Trainer(
                    {"generator": {"name": DCGANGenerator, "args": {"out_channels": 1, "step_channels":
                                   16}, "optimizer": {"name": Adam, "args": {"lr": 0.0002,
                                   "betas": (0.5, 0.999)}}},
                     "discriminator": {"name": DCGANDiscriminator, "args": {"in_channels": 1,
                                       "step_channels": 16}, "optimizer": {"var": "opt_discriminator",
                                       "name": Adam, "args": {"lr": 0.0002, "betas": (0.5, 0.999)}}}},
                    [MinimaxGeneratorLoss(), MinimaxDiscriminatorLoss()],
                    sample_size=64, epochs=20)
    """

    def __init__(
        self,
        models,
        losses_list,
        metrics_list=None,
        device=torch.device("cuda:0"),
        ncritic=10,
        epochs=5,
        sample_size=8,
        checkpoints="./model/gan",
        retain_checkpoints=None,
        recon="./images",
        log_dir=None,
        test_noise=None,
        nrow=8,
        verbose=True,
        tune_report=None,
        **kwargs
    ):
        super(ProximalTrainer, self).__init__(
            losses_list,
            metrics_list=metrics_list,
            device=device,
            ncritic=ncritic,
            epochs=epochs,
            sample_size=sample_size,
            checkpoints=checkpoints,
            retain_checkpoints=retain_checkpoints,
            recon=recon,
            log_dir=log_dir,
            test_noise=test_noise,
            nrow=nrow,
            **kwargs
        )
        self.model_names = []
        self.optimizer_names = []
        self.schedulers = []
        for key, model in models.items():
            self.model_names.append(key)
            if "args" in model:
                setattr(self, key, (model["name"](**model["args"])).to(self.device))
                if("discriminator" in key):
                    setattr(self, "proximal_"+key, (model["name"](**model["args"])).to(self.device))
            else:
                setattr(self, key, (model["name"]()).to(self.device))
                if("discriminator" in key):
                    setattr(self, "proximal_"+key, (model["name"](**model["args"])).to(self.device))
            opt = model["optimizer"]
            opt_name = "optimizer_{}".format(key)
            if "var" in opt:
                opt_name = opt["var"]
            self.optimizer_names.append(opt_name)
            model_params = getattr(self, key).parameters()
            if "args" in opt:
                setattr(self, opt_name, (opt["name"](model_params, **opt["args"])))
            else:
                setattr(self, opt_name, (opt["name"](model_params)))
            if "scheduler" in opt:
                sched = opt["scheduler"]
                if "args" in sched:
                    self.schedulers.append(
                        sched["name"](getattr(self, opt_name), **sched["args"])
                    )
                else:
                    self.schedulers.append(sched["name"](getattr(self, opt_name)))

        self.logger = Logger(
            self,
            losses_list,
            metrics_list,
            log_dir=log_dir,
            nrow=nrow,
            test_noise=test_noise,
            verbose = verbose
        )
        self.verbose = verbose
        self._store_loss_maps()
        self._store_metric_maps()
        self.tune_report=tune_report
    
    def train_iter(self):
        r"""Calls the train_ops of the loss functions. This is the core function of the Trainer. In most
        cases you will never have the need to extend this function. In extreme cases simply extend
        ``train_iter_custom``.

        .. warning::
            This function is needed in this exact state for the Trainer to work correctly. So it is
            highly recommended that this function is not changed even if the ``Trainer`` is subclassed.

        Returns:
            An NTuple of the ``generator loss``, ``discriminator loss``, ``number of times the generator
            was trained`` and the ``number of times the discriminator was trained``.
        """
        self.train_iter_custom()
        ldis, lgen, dis_iter, gen_iter = 0.0, 0.0, 0, 0
        loss_logs = self.logger.get_loss_viz()
        grad_logs = self.logger.get_grad_viz()

        for name, loss in self.losses.items():
            if isinstance(loss, GeneratorLoss) and isinstance(loss, DiscriminatorLoss):
                # NOTE(avik-pal): In most cases this loss is meant to optimize the Discriminator
                #                 but we might need to think of a better solution
                if self.loss_information["generator_iters"] % self.ngen == 0:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )
                    loss_logs.logs[name].append(cur_loss)
                    if type(cur_loss) is tuple:
                        lgen, ldis, gen_iter, dis_iter = (
                            lgen + cur_loss[0],
                            ldis + cur_loss[1],
                            gen_iter + 1,
                            dis_iter + 1,
                        )
                    else:
                        # NOTE(avik-pal): We assume that it is a Discriminator Loss by default.
                        ldis, dis_iter = ldis + cur_loss, dis_iter + 1
                for model_name in self.model_names:
                    grad_logs.update_grads(model_name, getattr(self, model_name))
            elif isinstance(loss, GeneratorLoss):
                # if self.loss_information["discriminator_iters"] % self.ncritic == 0:
                for _ in range(self.ngen):
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )
                    loss_logs.logs[name].append(cur_loss)
                    lgen, gen_iter = lgen + cur_loss, gen_iter + 1
                for model_name in self.model_names:
                    model = getattr(self, model_name)
                    if isinstance(model, Generator):
                        grad_logs.update_grads(model_name, model)
            elif isinstance(loss, DiscriminatorLoss):
                self.proximal_discriminator.load_state_dict(self.discriminator.state_dict())
                for _ in range(self.ncritic):
                # if self.loss_information["generator_iters"] % self.ngen == 0:
                    cur_loss = loss.train_ops(
                        **self._get_arguments(self.loss_arg_maps[name])
                    )
                    loss_logs.logs[name].append(cur_loss)
                    ldis, dis_iter = ldis + cur_loss, dis_iter + 1
                for model_name in self.model_names:
                    model = getattr(self, model_name)
                    if isinstance(model, Discriminator):
                        grad_logs.update_grads(model_name, model)
        return lgen, ldis, gen_iter, dis_iter
