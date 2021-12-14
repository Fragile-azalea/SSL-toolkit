[![Documentation Status](https://readthedocs.org/projects/ssl-toolkit/badge/?version=latest)](https://ssl-toolkit.readthedocs.io/en/latest/?badge=latest) ![](https://img.shields.io/badge/license-MIT-blue)

# DeSSL: A Pytorch toolkit for Deep Semi-Supervised Learning


## Requirements

```consule
python=3.8
```

## Description

DeSSL is an open source Python toolkit for deep semi-supervised learning.

The toolkit is based on PyTorch with high performance and friendly API.

Our code is pythonic, and the design is consistent with torchvision. You can easily develop new algorithms, or readily apply existing algorithms.

## Installation 

For flexible use and modification, please git clone the library.

```console
git clone https://github.com/Fragile-azalea/SSL-toolkit.git
cd SSL-toolkit
pip install -r doc/requirements.txt
```

### test

To check the integrity of the DeSSL installation
```console
cd test
python dataset_download.py --DATASET_PATH
pytest .
```

## Documentation

You can find the tutorial and API documentation on the website: [DeSSL Documentation](https://ssl-toolkit.readthedocs.io/en/latest/) .

We have examples in the directory [examples](https://github.com/Fragile-azalea/SSL-toolkit/tree/main/example).  A example usage is 

```python
......
def main(args):
    # Section 1: instantiate semi-labeled dataset
    mnist = SEMI_DATASET_REGISTRY(args.dataset)(args.root, 10)
    # Section 2: instantiate dataloader
    train_loader, test_loader, num_classes = mnist(
        args.batch_size, num_workers=args.num_workers) 
    # Section 2: define optimizer
    optimizer = {'optimizer': Adam, 'lr': args.lr_256 * args.batch_size / 256}
    # Section 3: define learn rate
    lr_scheduler = {'lr_scheduler': LambdaLR, 'lr_lambda': lambda epoch: epoch *
                    0.18 + 0.1 if epoch < 5 else (1. if epoch < 50 else 1.5 - epoch / 100)}
    # Section 4: instantiate model
    lenet = MODEL_REGISTRY(args.model)(**vars(args))
    # Section 5: instantiate trainer
    ladder = Ladder((train_loader, test_loader), optimizer,
                    lr_scheduler, lenet, args.lam_list)
    # Section 6: training !
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(ladder)
```

In the directory [examples](https://github.com/Fragile-azalea/SSL-toolkit/tree/main/example), you can find all the necessary running scripts to reproduce the benchmarks with specified hyperparameters.

## Contact

If you have any problem with our code or have some suggestions, including the future feature, feel free to contact 

Xiangli Yang (xlyang@std.uestc.edu.cn)

Xinglin Pan

or describe it in Issues.

## Citation

Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) [Computer software]. https://doi.org/10.5281/zenodo.3828935




## Acknowledgment






