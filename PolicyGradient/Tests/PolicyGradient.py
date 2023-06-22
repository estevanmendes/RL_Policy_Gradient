from PolicyGradient.PolicyGradient import PolicyGradient


def test():
    PG=PolicyGradient(1,1,1,1,1)
    print(f'{str(PG)} Tested')

def lunar_landing_test():
    import keras 
    import gym
    import numpy as np

    class PG_model:
        def __init__(self) -> None:
            model=keras.Sequential()
            model.add(keras.layers.Dense(10,input_shape=(8,)))
            model.add(keras.layers.Dense(4,activation='softmax'))
            self.model=model
           
    env=gym.make('LunarLander-v2')
    nn=PG_model()

    loss=keras.losses.categorical_crossentropy
    factor=0.99
    gradient_update_period=40
    learning_rate=0.02
    optimizer=keras.optimizers.Adam(learning_rate)
    RL=PolicyGradient(env=env,nn=nn,loss_fn=loss,min_score_aceptable=-300,state_size=8,metrics_display_period=3)
    scores,_,_=RL.training_protocol(6,optimizer=optimizer,gradient_update_period=10,factor=0.98)