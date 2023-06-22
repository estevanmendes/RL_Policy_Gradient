from PolicyGradient.PolicyGradientParalel import PolicyGradient


def test():
    PG=PolicyGradient(1,1,1,1,1,1,1)
    print(f'{str(PG)} Tested')

def lunar_landing_test():
    import keras 
    import gym
    import numpy as np

    model=keras.Sequential()
    model.add(keras.layers.Dense(10,input_shape=(8,)))
    model.add(keras.layers.Dense(4,activation='softmax'))
           
    env=gym.make('LunarLander-v2')
    nn=model

    loss=keras.losses.categorical_crossentropy
    factor=0.99
    gradient_update_period=40
    learning_rate=0.02
    optimizer=keras.optimizers.Adam(learning_rate)
    RL=PolicyGradient(n_cpus=2,env=env,nn=nn,loss_fn=loss,gradient_update_period=3,min_score_aceptable=-300,state_size=8,metrics_display_period=3)
    scores,_,_=RL.training_protocol(6,optimizer=optimizer,factor=0.98)
    RL.display_score(scores,2)