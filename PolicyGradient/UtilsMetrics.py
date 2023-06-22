import os 
import numpy as np



def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def save_results_for_latter_comparison(factor,gradient_update_period,lr):
        os.rename('scores.txt',f'scores_fator_{factor}_periodo_gradiente_{gradient_update_period}_lr_{lr}_.txt')


def describe(data):
    print(f'mean:{np.mean(data)}\nmax:{np.max(data)}\nmin:{np.min(data)}')


def metrics_decorator(report_period=20):

    def metrics_(f):
        def wrapper(*args,**kwargs):
            global counter
            if counter%report_period==0:
                print(f'{counter} episodes')
            counter+=1
            return f(*args,**kwargs)
        return wrapper
    return metrics_