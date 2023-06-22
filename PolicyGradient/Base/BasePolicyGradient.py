import numpy as np
from tensorflow import GradientTape,reduce_mean,Tensor
from abc import abstractclassmethod
import keras
from PolicyGradient.Utils import UtilsRewards
from PolicyGradient.Utils import UtilsGrads
from PolicyGradient.Utils import UtilsMetrics
from typing import Any, Tuple,List


class BasePolicyGradient():
    @staticmethod
    def discount_reward_episode(*args,**kwargs):
        return UtilsRewards.discount_reward_episode(*args,**kwargs)
    
    @staticmethod
    def discount_rewards_multiple_episodes(*args,**kwargs):
        return UtilsRewards.discount_rewards_multiple_episodes(*args,**kwargs)

    @staticmethod
    def normalize_vector(*args,**kwargs):
        return UtilsRewards.normalize_vector(*args,**kwargs)

    @staticmethod
    def normalize_matrix(*args,**kwargs):
        return UtilsRewards.normalize_matrix(*args,**kwargs)

    @staticmethod
    def apply_reward_weight_in_grads(*args,**kwargs):
        return UtilsRewards.apply_reward_weight_in_grads(*args,**kwargs)

    @staticmethod
    def average_grads_from_steps(*args,**kwargs):
        return UtilsGrads.average_grads_from_steps(*args,**kwargs)
    
    @staticmethod
    def average_grads_from_episodes(*args,**kwargs):
        return UtilsGrads.average_grads_from_episodes(*args,**kwargs)

    @staticmethod
    def apply_police_gradient(*args,**kwargs):
        return UtilsGrads.apply_police_gradient(*args,**kwargs)
    
    @staticmethod
    def display_score(*args,**kwargs):
        return UtilsMetrics.display_score(*args,**kwargs)



    def __init__(self,env,state_size,min_score_aceptable,nn,loss_fn:keras.losses,gradient_update_period,hold_results=False,metrics_display_period=10*4) -> None:
        """
        
        """
        self.env=env
        self.min_score_aceptable=min_score_aceptable
        self.state_size=state_size
        self.nn=nn
        self.loss_fn=loss_fn
        self.gradient_update_period=gradient_update_period

        self.hold_results=hold_results
        self.metrics_display_period=metrics_display_period
        self.grads=[]
        self.rewads=[]
        self.scores=[]
        self._counter=1


    @UtilsMetrics.metrics_method_decorator   
    def run_episode(self,method)->Tuple[List[float],List[float],List[List[Tensor]]]:
        """
        model:
        loss_fn:
        method:
        
        """
        
        state=self.env.reset()
        score=0
        rewards=[]
        grads=[]
        while True:
                action,step_grads=self.action_and_grads(state,self.nn,method=method)
                state,reward,done,info=self.env.step(action)
                score+=reward
                rewards.append(reward)
                grads.append(step_grads)
                if done or score<self.min_score_aceptable:      
                    with open('scores.txt','a+') as f:
                        f.write(f'{score}\n')  

                    break

        return score,rewards,grads
    
    def run_episode_paralel(self,method,env)->Tuple[List[float],List[float],List[List[Tensor]]]:
        """
        model:
        loss_fn:
        method:
        
        """
        
        state=env.reset()
        score=0
        rewards=[]
        grads=[]
        while True:
                action,step_grads=self.action_and_grads(state,self.nn,method=method)
                state,reward,done,info=env.step(action)
                score+=reward
                rewards.append(reward)
                grads.append(step_grads)
                if done or score<self.min_score_aceptable:      
                    with open('scores.txt','a+') as f:
                        f.write(f'{score}\n')  

                    break
                

        return score,rewards,grads
    



    def action_and_grads(self,state,nn,method='roulette_prob')-> Tuple[int, List]:
        """
        method:["max_prob","roulette_prob"]

        max_prob: It chooses the action with the highest probability
        roulette_prob: It chooses the action randomly, where the probablity of each action are designed by the model
        
        """
        state=np.array(state).reshape(-1,self.state_size)

        if method=="max_prob":
            with GradientTape() as tape:
                probs=nn(state)## method for action
                action=np.argmax(probs)
                target=np.zeros(probs.shape)
                target[0][action]=1.
                loss=reduce_mean(self.loss_fn(target,probs))

        elif method=="roulette_prob":
            with GradientTape() as tape:
                probs=nn(state)## method for action
                action=np.random.choice(np.arange(probs.shape[1]),p=probs.numpy().flatten(),size=1)[0]
                target=np.zeros(probs.shape)
                target[0][action]=1.
                loss=reduce_mean(self.loss_fn(target,probs))

        grads=tape.gradient(loss,nn.trainable_variables)
        return action,grads
    
    @abstractclassmethod
    def run_multiple_episodes(self):
        pass
    


    @abstractclassmethod
    def training_protocol(self):
        pass

    def __str__(self) -> str:
        return 'BasePolicyGradient'
