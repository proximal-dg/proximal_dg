#--------- CONVERGENCE ------------

#---------SNGAN------------
# bash deploy.sh SNGAN/MNIST/convergence/train . train.py 2017csb1104 sahil_pytorch  '{"model":"SNGAN","loss_type":"minimax","dataset":"MNIST","gpu":0,"channels":1,"step_channels":32,"epochs":200,"resume":0}' 1
bash deploy.sh SNGAN/CIFAR-10/convergence/train . train.py 2017csb1104 sahil_pytorch  '{"model":"SNGAN","loss_type":"minimax","dataset":"CIFAR-10","epochs":100,"gpu":0,"resume":1}' 1
# bash deploy.sh SNGAN/CELEB-A/convergence/train . train.py 2017csb1104 sahil_pytorch  '{"model":"SNGAN","loss_type":"minimax","dataset":"CELEB-A"}' 1


#---------WGGAN------------
# bash deploy.sh WGAN-WC/MNIST/convergence/train . train.py 2017csb1104 sahil_pytorch  '{"model":"WGAN-WC","loss_type":"wgan","dataset":"MNIST","gpu":0,"channels":1,"step_channels":32,"epochs":200,"resume":0}' 1
# bash deploy.sh WGAN-WC/CIFAR-10/convergence/train . train.py 2017csb1104 sahil_pytorch  '{"model":"WGAN-WC","loss_type":"wgan","dataset":"CIFAR-10","gpu":0,"channels":3,"step_channels":64,"epochs":100,"resume":0}' 1
# bash deploy.sh WGAN-WC/CELEB-A/convergence/train . train.py 2017csb1104 sahil_pytorch  '{"model":"WGAN-WC","loss_type":"wgan","dataset":"CELEB-A","gpu":0,"channels":3,"step_channels":64,"epochs":100,"resume":0}' 1


#---------SAGAN------------
# bash deploy.sh SAGAN/MNIST/convergence/train . train.py 2017csb1104 sahil_pytorch '{"model":"SAGAN","loss_type":"minimax","dataset":"MNIST","gpu":0,"channels":1,"step_channels":16,"epochs":100,"resume":0}'
# bash deploy.sh SAGAN/CIFAR-10/convergence/train . train.py 2017csb1104 sahil_pytorch '{"model":"SAGAN","loss_type":"minimax","dataset":"CIFAR-10","gpu":0,"channels":3,"step_channels":32}'
# bash deploy.sh SAGAN/CELEB-A/convergence/train . train.py 2017csb1104 sahil_pytorch '{"model":"SAGAN","loss_type":"minimax","dataset":"CELEB-A","gpu":0,"channels":3}'