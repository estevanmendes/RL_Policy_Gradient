import os 
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt
from PolicyGradient.Utils import UtilsSaving

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def describe(data):
    print(f'mean:{np.mean(data)}\nmax:{np.max(data)}\nmin:{np.min(data)}')


def metrics_decorator(report_period=20):
    def metrics_(f):
        global counter
        def wrapper(*args,**kwargs):
            if counter%report_period==0:
                print(f'{counter} episodes')
            counter+=1
            return f(*args,**kwargs)
        return wrapper
    return metrics_


def metrics_method_decorator(f):
        def wrapper(self,*args,**kwargs):
            if self._counter%self.metrics_display_period==0:
                print(f'{self._counter} episodes')
            self._counter+=1
            return f(self,*args,**kwargs)
        return wrapper

def display_score(scores,window):
    """
    It plots a graph of the score, i.e. the sum of rewards, curve by the episodes. It is applied a moving average to smooth the curve.

    Parameters
    ----------
    scores:list,array
        The list of float or ints containing the historical serie of scores
    window: int
        the window to apply a moving average. 
    """
    plt.plot(np.arange(len(scores)-window+1),moving_average(scores,window))
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.grid(True)
    plt.show()

def check_convergence(f):
        def wrapper(self,*args,**kwargs):
            scores,rewards,grads=f(self,*args,**kwargs)
            median=np.median(scores[-self.threshold_window:])    
            if median>=self.threshold:
                self.nn.save('models/model_100%_of_threshold_score_{self.threshold}')

            elif median>=0.5*self.threshold:
                 self.nn.save('models/model_50%_of_threshold_score_{self.threshold}')
                 
                    
        return wrapper
