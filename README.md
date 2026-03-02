# GmNet: Revisiting Gating Mechanisms From A Frequency View

<p align="center"> <b>ICLR 2026</b> </p> <p align="center"> <a href="https://arxiv.org/abs/2503.22841">📄 arxiv</a> | <a href="https://github.com/YFWang1999/GmNet">💻 Code</a> </p>

### Install requirements

Run the following command to install the dependences:

```bash
pip install -r requirements.txt
```

### Data preparation

We need to prepare ImageNet-1k dataset from [`http://www.image-net.org/`](http://www.image-net.org/).

- ImageNet-1k

ImageNet-1k contains 1.28 M images for training and 50 K images for validation.
The images shall be stored as individual files:

```
ImageNet/
├── train
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
...
├── val
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
...
```

Our code also supports storing the train set and validation set as the `*.tar` archives:

```
ImageNet/
├── train.tar
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
...
└── val.tar
│   ├── n01440764
│   │   ├── ILSVRC2012_val_00000293.JPEG
...
```



## Training

To train the model on a single node with 8 GPUs for 300 epochs and distributed evaluation, run:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 train_imagenet.py --data {path to dataset} --model gmnet_s3 -b 256 --lr 3e-3 --weight-decay 0.05 --aa rand-m1-mstd0.5-inc1 --cutmix 0.2 --color-jitter 0. --drop-path 0. --log-wandb
```



## Speed test

Run the following command to compare the throughputs on GPU/CPU:

```bash
python benchmark_onnx.py.py
```

## BibTeX

    @article{wang2025gmnet,
        title={GmNet: Revisiting Gating Mechanisms From A Frequency View},
        author={Wang, Yifan and Ma, Xu and Zhang, Yitian and Wang, Zhongruo and Kim, Sung-Cheol and Mirjalili, Vahid and Renganathan, Vidya and Fu, Yun},
        journal={arXiv preprint arXiv:2503.22841},
        year={2025}
        }

## License
The majority of GmNet is licensed under an [Apache License 2.0](https://github.com/ma-xu/Rewrite-the-Stars/blob/main/LICENSE)


