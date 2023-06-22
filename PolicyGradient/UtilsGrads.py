import numpy as np
from tensorflow import reduce_mean,keras

def average_grads_from_steps(nn,grads):
    averaged_grads=[]
    for variable_index in range(len(nn.model.trainable_variables)):
        grad_steps_from_variable=[]
        for grad in grads:
            grad_steps_from_variable.append(grad[variable_index])

        average_grad_in_episode=reduce_mean(grad_steps_from_variable,axis=0)
        averaged_grads.append(average_grad_in_episode)
         
    return averaged_grads



def average_grads_from_episodes(nn,grads):
    averaged_grads=[]
    for variable_index in range(len(nn.model.trainable_variables)):
        grad_steps_from_variable=[]
        for grad_episode in grads:
            for grad_step in grad_episode:
                grad_steps_from_variable.append(grad_step[variable_index])

        average_grad_in_episode=reduce_mean(grad_steps_from_variable,axis=0)
        averaged_grads.append(average_grad_in_episode)
         
    return averaged_grads

def apply_police_gradient(optimizer:keras.optimizers,grad,nn):

        optimizer.apply_gradients(zip(grad,nn.model.trainable_variables))