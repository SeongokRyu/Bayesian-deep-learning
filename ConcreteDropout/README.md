# Concrete dropout 

* The copyright of code "ConcreteDropout.py" to Yarin Gal, https://github.com/yaringal/ConcreteDropout 
* Paper: Yarin Gal, Jiri Hron, Alex Kendall, Concrete Dropout, NIPS 2017
* I fixed some commands in order to successfully run at Tensorflow version >1.10. (The original code at Yarin's github didn't work at TF v1.10)


# Usage
## 1. First, make 'data', 'figures' and 'statistics' directories to save input, figures and output files.

> mkdir data

> mkdir figures

> mkdir statistics

## 2. Training: 

python train.py 'hidden_dim' '# of training samples' '# of test samples' 'amount of noise' 'length of gaussian prior' '# of epoch'
For example,
> python train.py 512 1000 1000 1.0 0.001 100
