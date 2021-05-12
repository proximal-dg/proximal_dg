## Characterizing GAN Convergence Through Proximal Duality Gap

Implementation of **ProximalDualityGap** | [Paper](https://arxiv.org/abs/2105.04801) ( ICML 2021 )

Sahil Sidheekh<sup>1</sup>, Aroof Aimen<sup>1</sup>, Narayanan C. Krishnan<sup>1</sup>

<sup>1</sup> <sub>Indian Institute of Technology, Ropar</sub>  

Despite the accomplishments of **Generative Adversarial Networks** (GANs) in modeling data distributions, training them remains a challenging task. A contributing factor to this difficulty is the non-intuitive nature of the GAN loss curves, which necessitates a subjective **evaluation** of the generated output to infer training progress. Recently, motivated by game theory, **duality gap** has been proposed as a domain agnostic measure to monitor GAN training. However, it is restricted to the setting when the GAN converges to a Nash equilibrium. But *GANs need not always converge to a Nash equilibrium to model the data distribution*. In this work, we extend the notion of duality gap to **proximal duality gap** that is applicable to the general context of training GANs where Nash equilibria may not exist. We show theoretically that the proximal duality gap is capable of monitoring the convergence of GANs to a wider spectrum of equilibria that subsumes Nash equilibria. We also theoretically establish the relationship between the proximal duality gap and the divergence between the real and generated data distributions for different GAN formulations. Our results provide new insights into the nature of GAN convergence.

<!-- <a href="http://www.youtube.com/watch?feature=player_embedded&v=lkjMxZDGubA
" target="_blank"><img src="http://img.youtube.com/vi/lkjMxZDGubA/0.jpg" 
alt="VIDEO" width="700" border="10" /></a> -->

## How to run

Our implementation is based on the [torchgan](https://github.com/torchgan/torchgan) framework and supports GAN architectures: *WGAN*,*SNGAN* over *MNIST*, *CIFAR10* and *CELEB-A* datasets.

#### To train :
```
python train.py --arg '{"model":"WGAN-WC","loss_type":"wgan","dataset":"CIFAR-10","channels":3,"img_size":32,"step_channels":32,"epochs":200,"ncritic":1,"resume":1,"gpu":0,"sample_size":100,"nrow":10}'
```

#### To evaluate :
```
python evaluate.py --arg '{"model":"WGAN-WC","loss_type":"wgan","dataset":"CIFAR-10","channels":3,"img_size":32,"step_channels":32,"epochs":200,"ncritic":5,"gpu":0,"sample_size":100,"nrow":10}'
```


## How to cite

If you find the code/theory for Proximal Duality Gap useful in your research, kindly consider citing the following paper.

```
@misc{sidheekh2021characterizing,
      title={Characterizing GAN Convergence Through Proximal Duality Gap}, 
      author={Sahil Sidheekh and Aroof Aimen and Narayanan C. Krishnan},
      year={2021},
      eprint={2105.04801},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```


## License

This project is distributed under [MIT license](LICENSE).

```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```


