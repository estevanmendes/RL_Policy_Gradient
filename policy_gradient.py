import multiprocessing
from itertools import chain
import copy 
import gym
import matplotlib.pyplot as plt
from gym.utils.play import play,PlayPlot
import pygame
import numpy as np
from scipy import stats
from tensorflow import keras
import tensorflow as tf
import os


def describe(data):
    print(f'mean:{np.mean(data)}\nmax:{np.max(data)}\nmin:{np.min(data)}')

def metrics(report_period=20):

    def metrics_(f):
        def wrapper(*args,**kwargs):
            global counter
            if counter%report_period==0:
                print(f'{counter} episodes')
            counter+=1
            return f(*args,**kwargs)
        return wrapper
    return metrics_

def run_episode(env,model,loss_fn,method,render=False):
    state=env.reset()
    score=0
    rewards=[]
    grads=[]
    if not render:
        while True:
            action,step_grads=action_and_grads(state,model,loss_fn,method=method)
            state,reward,done,info=env.step(action)
            score+=reward
            rewards.append(reward)
            grads.append(step_grads)
            if done or score<-300:      
                with open('training.txt','a+') as f:
                    f.write('-----------------------------------------\n')
                    f.write(f'reward: {score}\n')  
                with open('scores.txt','a+') as f:
                    f.write(f'{score}\n')  

                break
    if render:
        while True:
            action,step_grads=action_and_grads(state,model,loss_fn,method=method)
            state,reward,done,info=env.step(action)
            score+=reward
            rewards.append(reward)
            grads.append(step_grads)
            if done or score<-300:      
                break

    return score,rewards,grads


def action_and_grads(state,nn,loss_fn,method='roulette_prob'):
    """
    method:["max_prob","roulette_prob"]
    
    """
    state=np.array(state).reshape(-1,8)

    if method=="max_prob":
        with tf.GradientTape() as tape:
            probs=nn.model(state)## method for action
            action=np.argmax(probs)
            target=np.zeros(probs.shape)
            target[0][action]=1.
            loss=tf.reduce_mean(loss_fn(target,probs))

    elif method=="roulette_prob":
        with tf.GradientTape() as tape:
            probs=nn.model(state)## method for action
            action=np.random.choice(np.arange(probs.shape[1]),p=probs.numpy().flatten(),size=1)[0]
            target=np.zeros(probs.shape)
            target[0][action]=1.
            loss=tf.reduce_mean(loss_fn(target,probs))

    grads=tape.gradient(loss,nn.model.trainable_variables)
    return action,grads


def run_multiple_episodes(number_of_episodes,env,nn,loss_fn,method='roulette_prob'):
    scores=[]
    rewards=[]
    grads=[]
    for i in range(number_of_episodes):
        score,epsisode_rewards,episode_grads=run_episode(env,nn,loss_fn,method)
        scores.append(score)
        rewards.append(epsisode_rewards)
        grads.append(episode_grads)
    return scores,rewards,grads

def run_episodes_in_paralel(n_cpus, number_of_episodes,env,model,action_fn,method):
    
    with multiprocessing.Pool(n_cpus) as pool:
        results=[pool.apply(run_episode,args=(copy.deepcopy(env),model,action_fn,method)) for i in range(number_of_episodes)]
    scores,rewards,grads=[],[],[]
    for (ep_scores,ep_rewards,ep_grads) in results:
        scores.append(ep_scores)
        rewards.append(ep_rewards)
        grads.append(ep_grads)
    return scores,rewards,grads


def discount_reward_episode(rewards,factor):
    
    def matrix_of_discount(number_of_vectors):
        """creates a matrix of the shape
        [[1,k,k^2,k^3],
         [1,1,k  ,k^2],
         [1,1,1  ,k  ],
         [1,1,1  ,1  ]
        ]
        """
        def vector_i(i):
            """it creates a vector of the shape [j^3,j^2,j^1,1,1,1,1], where the i designates the max order of the vector, also i=<number_of_vectors"""
            vector=np.array([factor**max(0,i-k) for k in range(number_of_vectors)])
            return vector
                
        matrix=np.zeros((number_of_vectors,number_of_vectors))
        for i in range(number_of_vectors):
            matrix[:,i]=vector_i(i)
        
        return matrix

    number_of_vectors=len(rewards)
    matrix_discount=matrix_of_discount(number_of_vectors)
    rewards=np.array(rewards)
    result=np.matmul(rewards,matrix_discount)
    return result   

def discount_rewards_multiple_episodes(rewards,factor):
    return np.array([discount_reward_episode(episode_rewards,factor) for episode_rewards in rewards])
 

def normalize_vector(vector):
    mean=np.mean(vector)
    std=np.std(vector)
    new_vector=(vector-mean)/std
    return new_vector

def normalize_matrix(matrix):
    flatten=np.fromiter(chain.from_iterable(matrix),dtype='float')
    mean=np.mean(flatten)
    std=np.std(flatten)
    new_matrix=(matrix-mean)/std
    return new_matrix

def apply_reward_weight_in_grads(rewards,grads):
    reward_matrix=np.tile(rewards,(grads.shape[1],1)).transpose()
    return reward_matrix*grads
    
def apply_police_gradient(optimizer:keras.optimizers,grad,nn):

        optimizer.apply_gradients(zip(grad,nn.model.trainable_variables))


def average_grads_from_steps(nn,grads):
    averaged_grads=[]
    for variable_index in range(len(nn.model.trainable_variables)):
        grad_steps_from_variable=[]
        for grad in grads:
            grad_steps_from_variable.append(grad[variable_index])

        average_grad_in_episode=tf.reduce_mean(grad_steps_from_variable,axis=0)
        averaged_grads.append(average_grad_in_episode)
         
    return averaged_grads



def average_grads_from_episodes(nn,grads):
    averaged_grads=[]
    for variable_index in range(len(nn.model.trainable_variables)):
        grad_steps_from_variable=[]
        for grad_episode in grads:
            for grad_step in grad_episode:
                grad_steps_from_variable.append(grad_step[variable_index])

        average_grad_in_episode=tf.reduce_mean(grad_steps_from_variable,axis=0)
        averaged_grads.append(average_grad_in_episode)
         
    return averaged_grads

def training_protocol(number_of_episodes,env,nn,loss_fn,optimizer,gradient_update_period=1,factor=0.5):
    scores=[]
    rewards=[]
    grads=[]
    method='roulette_prob'
    for episode in range(1,number_of_episodes+1):
        score,episode_rewards,episode_grads=run_episode(env,nn,loss_fn,method)
        episode_rewards=discount_reward_episode(episode_rewards,factor)
        episode_grads=np.asarray(episode_grads)
        scores.append(score)
        rewards.append(episode_rewards)
        grads.append(episode_grads)

        

        if episode%gradient_update_period==0:
            batch_rewards=rewards[-gradient_update_period:]
            batch_grads=grads[-gradient_update_period:]
            batch_rewards=normalize_matrix(batch_rewards)
            batch_grads=[apply_reward_weight_in_grads(episode_rewards,episode_grads)  for episode_rewards,episode_grads in zip(batch_rewards,batch_grads)] 
            batch_grads=average_grads_from_episodes(nn,batch_grads)
            apply_police_gradient(optimizer,batch_grads,nn)  
            del batch_rewards
            del batch_grads
            del rewards
            del grads
            rewards=[]
            grads=[]


    return scores,rewards,grads

def training_protocol_paralel(n_cpus,number_of_episodes,env,nn,loss_fn,optimizer,gradient_update_period=1,factor=0.5, threshold=150):
    scores=[]
    rewards=[]
    grads=[]
    method='roulette_prob'
    changed=False

    updates=number_of_episodes//gradient_update_period+int(bool(number_of_episodes%gradient_update_period))
    for policy_update in range(1,updates+1):
        episodes_scores,batch_rewards,batch_grads=run_episodes_in_paralel(n_cpus,gradient_update_period,env,nn,loss_fn,method=method)
        batch_rewards=[discount_reward_episode(episode_rewards,factor) for episode_rewards in batch_rewards ]
        batch_grads=[np.asarray(episode_grads)for episode_grads in batch_grads]
        scores.extend(episodes_scores)
        rewards.extend(batch_rewards)
        grads.extend(batch_grads)
        

        batch_rewards=normalize_matrix(batch_rewards)
        batch_grads=[apply_reward_weight_in_grads(episode_rewards,episode_grads)  for episode_rewards,episode_grads in zip(batch_rewards,batch_grads)] 
        batch_grads=average_grads_from_episodes(nn,batch_grads)
        apply_police_gradient(optimizer,batch_grads,nn)  
        del batch_rewards
        del batch_grads
        del rewards
        del grads
        rewards=[]
        grads=[]
        if np.median(scores[-gradient_update_period:])>threshold*.5 and not changed:                
            lr=keras.backend.get_value(optimizer.learning_rate)
            keras.backend.set_value(optimizer.learning_rate, lr*0.1)
            nn.model.save(f'model_threshold_50%_{threshold}.h5')
            print(f'learning rate changed from {lr} to {lr*0.1}')
            changed=True

        if np.median(scores[-gradient_update_period:])>threshold:
                nn.model.save(f'model_threshold_{threshold}.h5')
                break



    return scores,rewards,grads
    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class PG_model:
    def __init__(self) -> None:
        model=keras.Sequential()
        model.add(keras.layers.Dense(10,input_shape=(8,)))
        model.add(keras.layers.Dense(5))
        model.add(keras.layers.Dense(4,activation='softmax'))
        self.model=model


    def action(self,state):
        state=np.array(state).reshape(-1,8)
        action=self.model(state)
        return action
    
    
np.random.seed(42)
env=gym.make('LunarLander-v2')
nn=PG_model()
nn.model=keras.models.load_model('model_threshold_50%_250.h5')

loss=keras.losses.categorical_crossentropy
lr=tf.keras.optimizers.schedules.PolynomialDecay(0.1,10**4,end_learning_rate=10**-5)

factor=0.99
gradient_update_period=40
learning_rate=0.02

optimizer=keras.optimizers.Adam(learning_rate)
iterations='11k'

scores,rewards,grads=training_protocol_paralel(7,3000,env,nn,loss,optimizer,gradient_update_period=gradient_update_period,factor=factor,threshold=250)
nn.model.save('model_14k.h5')

scores,rewards,grads=training_protocol_paralel(7,3000,env,nn,loss,optimizer,gradient_update_period=gradient_update_period,factor=factor,threshold=250)
nn.model.save('model_17k.h5')

scores,rewards,grads=training_protocol_paralel(7,3000,env,nn,loss,optimizer,gradient_update_period=gradient_update_period,factor=factor,threshold=250)
nn.model.save('model_14k.h5')


def save_results_for_latter_comparison(factor,gradient_update_period,lr):
        os.rename('scores.txt',f'scores_fator_{factor}_periodo_gradiente_{gradient_update_period}_lr_{lr}_.txt')

save_results_for_latter_comparison(factor,gradient_update_period,learning_rate)
nn.model.save(f'iterations_{iterations}_fator_{factor}_periodo_gradiente_{gradient_update_period}_lr_{learning_rate}.h5')


