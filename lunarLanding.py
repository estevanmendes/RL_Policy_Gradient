from tensorflow import keras
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PolicyGradient.PolicyGradient import PolicyGradient as PG
from PolicyGradient.PolicyGradientParalel import PolicyGradient as PGP



# model=keras.Sequential()
# model.add(keras.layers.Dense(10,input_shape=(8,)))
# model.add(keras.layers.Dense(5))
# model.add(keras.layers.Dense(4,activation='softmax'))

np.random.seed(42)
env=gym.make('LunarLander-v2')
# nn=model

loss=keras.losses.categorical_crossentropy
factor=0.98
gradient_update_period=40
learning_rate=0.02
optimizer=keras.optimizers.Adam(learning_rate)
env.reset()
# state,reward,done,info=
state=env.step(1)
state=env.step(2)

print(state[0])
# RL=PG(env=env,nn=nn,loss_fn=loss,gradient_update_period=10,min_score_aceptable=-300,state_size=8,metrics_display_period=100)
# scores,_,_=RL.training_protocol(1500,optimizer=optimizer,factor=0.98)
# RL.display_score(scores,30)
# RL.display_score(scores,30)
# scores,_,_=RL.training_protocol(1501,optimizer=optimizer,factor=0.98)
# RL.display_score(scores,30)
# RL.save_trainning(factor,learning_rate)