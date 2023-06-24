from tensorflow import keras
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PolicyGradient.PolicyGradient import PolicyGradient as PG
from PolicyGradient.PolicyGradientParalel import PolicyGradient as PGP
import keras_cv


model=keras.Sequential()
model.add(keras_cv.layers.Grayscale(1,input_shape=(96,96,3)))
model.add(keras.layers.Conv2D(filters=30,kernel_size=(6,6),strides=3))##(16(96-6/6+1),31,10)
model.add(keras.layers.MaxPool2D(pool_size=(3,3),strides=3))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(5,activation='softmax'))

np.random.seed(42)
env=gym.make('CarRacing-v2',continuous=False)



nn=model
    
loss=keras.losses.categorical_crossentropy
factor=0.99
learning_rate=0.2
optimizer=keras.optimizers.Adam(learning_rate)
gradient_update_period=10

RL=PG(env=env,nn=nn,loss_fn=loss,gradient_update_period=gradient_update_period,min_score_aceptable=-50,state_size=(96,96,3),metrics_display_period=100)
scores,_,_=RL.training_protocol(1000,optimizer=optimizer,factor=factor)
RL.display_score(scores,10)
RL.save_trainning(factor,learning_rate)