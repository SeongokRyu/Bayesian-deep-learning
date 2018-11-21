# Concrete dropout 
### The copyright of code "ConcreteDropout.py" to Yarin Gal, https://github.com/yaringal/ConcreteDropout
### Yarin Gal, Jiri Hron, Alex Kendall, Concrete Dropout, NIPS 2017


# Usage
## 1. First, make 'data', 'figures' and 'statistics' directories to save input, figures and output files.

> mkdir data
> mkdir figures
> mkdir statistics

## 2. Training: 

python train.py 'hidden_dim' '# of training samples' '# of test samples' 'amount of noise' 'length of gaussian prior' '# of epoch'
> python train.py 512 1000 1000 1.0 0.001 100
