import os 
import numpy as np
from functools import wraps
import matplotlib.pyplot as plt


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def save_results_for_latter_comparison(factor,gradient_update_period,lr):
        os.rename('scores.txt',f'scores_fator_{factor}_periodo_gradiente_{gradient_update_period}_lr_{lr}_.txt')


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
        def error(self,*args,**kwargs):
            if self._counter%self.metrics_display_period==0:
                print(f'{self._counter} episodes')
            self._counter+=1
            return f(self,*args,**kwargs)
        return error

def display_score(scores,window):    
    plt.plot(np.arange(len(scores)-window+1),moving_average(scores,window))
    plt.ylabel('Score')
    plt.xlabel('Episodes')
    plt.grid(True)
    plt.show()