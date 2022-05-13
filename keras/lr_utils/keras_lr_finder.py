from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


##########################
## LEARNING RATE FINDER ##
##########################

class LRFinder(Callback):
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 step_size=np.ceil(epoch_size/batch_size), 
                                 beta=0.98)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        step_size: Number of mini-batches in the dataset. 
        beta: Parameter for averaging the loss. To pick between 0 and 1.
        
    # References
        Original paper: https://arxiv.org/abs/1506.01186
    '''

    def __init__(self, min_lr=1e-5, max_lr=1e-2, step_size=None, beta=0.98):
        super().__init__()

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.beta = beta
        self.step_size = step_size
        self.iteration = 0
        self.lr_mult = 0
        self.best_loss = 0.
        self.avg_loss = 0
        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        self.lr_mult = (self.max_lr / self.min_lr) ** (1 / self.step_size)
        return self.lr_mult

    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        # Smooth the loss
        loss = logs.get('loss')
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** self.iteration)

        # Check if the loss is not exploding
        if self.iteration > 1 and smoothed_loss > self.best_loss * 4:
            self.model.stop_training = True
            return

        if smoothed_loss < self.best_loss or self.iteration == 1:
            self.best_loss = smoothed_loss

        # Increase the learning rate
        lr = K.get_value(self.model.optimizer.lr) * self.clr()

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)
        self.history.setdefault('avg_loss', []).append(smoothed_loss)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, lr)

    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')

    def plot_avg_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'][10:-5], self.history['avg_loss'][10:-5])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Avg Loss')

    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'][10:-5], self.history['loss'][10:-5])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
