## Code for CIFAR-10/100 experiments of Training Thinner and Deeper Neural Networks: Jumpstart Regularization

### Structure of the code

The code is divided in three parts: the implementation of the engine to run the hyperparameter grids of the experiments, the config files storing the hyperparameters used and the actual code of the Jumpstart Regularization.

#### Implementation of the Jumpstart Regularization

The Jumpstart Regularization is implemented in the class JumpstartRegularization of the 
module jumpstart.loss. It captures the preactivations throught pytorch hooks and exposes the loss with the loss property, which can be added to the main loss at the main training loop or elsewhere.

#### Running the experiments

In order to run the experiments just call grid.py with the appropiate config. For instance, to run the CIFAR-10 experiments use:

``` python grid.py config.cifar_10.py``` 

This will log the resulting metrics locally and wandb to be used later. 

#### Generating the figures

In order to generate the figures, the easiest is to download the wandb data in a csv file and use it to generate the plots. To do so, just call ```python main.py download``` with the appropiate credentials to generate the csv files, and then call the different functions inside main.py which will generate the plots. For instance, to generate the plots for the CIFAR-10 experiment just call:

```python main.py plot_cifar_param_count```

