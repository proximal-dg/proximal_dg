# On Characterizing GAN Convergence Through Proximal Duality Gap

#### To train :
```
python train.py --arg '{"model":"WGAN-WC","loss_type":"wgan","dataset":"CIFAR-10","channels":3,"img_size":32,"step_channels":32,"epochs":200,"ncritic":1,"resume":1,"gpu":0,"sample_size":100,"nrow":10}'
```

#### To evaluate :
```  
python evaluate.py --arg '{"model":"WGAN-WC","loss_type":"wgan","dataset":"CIFAR-10","channels":3,"img_size":32,"step_channels":32,"epochs":200,"ncritic":5,"gpu":0,"sample_size":100,"nrow":10}'
```

