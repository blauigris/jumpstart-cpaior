## Code for Moons and MNIST experiments of Training Thinner and Deeper Neural Networks: Jumpstart Regularization

### Structure of the code

The code is divided in three parts: the implementation of the , 
the experiment code and the jupyter notebooks used to run the actual experiments of the papers.

#### Implementation of the Jumpstart Constraint

The Jumpstart Regularization is implemented in the class JumpstartReLU of the 
module jumpstart.py. It is meant like a regular activation layer from keras, 
see jumpstart_experiment.py for examples.

#### Experiment code

The code for performing a single experiment, this is a run with a set of parameters, is located 
at experiment.py. However, this class is meant to be overriden, so sep_cons_experiment.py contains
the actual code to build the networks used in the experiments. There are a couple of 
examples for Glorot and Zero initialization. There are many visualization options available
which enable (when work) to see the internal representations.

#### Jupyter notebooks

This is the actual code used to generate the experiments for the paper. It is divided 
into many different combinations of parameters and datasets in order to run them 
separately for the sake of parallelization. All of them generate a set of tensorboard summary
files which are parsed by the table*.ipynb notebooks to generate the plots.

#### Other code

Visualization code and an implementation of Annealed Dropout are stored inside callbacks.py. There
is also an implementation of CLR ad SGDR learning rate schemes. datasets package stores
the code to load and prepare the data for the experiments.

