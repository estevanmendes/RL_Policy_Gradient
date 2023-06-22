import PolicyGradient.Base.BasePolicyGradient as BPG #_PolicyGradient
import copy 
import keras
from typing import Tuple,List
import numpy as np

class PolicyGradient(BPG.BasePolicyGradient):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)

    def run_multiple_episodes(self,number_of_episodes,nn,loss_fn,method='roulette_prob')->Tuple[List,List,List]:
        """
        """
        scores=[]
        rewards=[]
        grads=[]
        for i in range(number_of_episodes):
            score,epsisode_rewards,episode_grads=self.run_episode(nn,loss_fn,method)
            scores.append(score)
            rewards.append(epsisode_rewards)
            grads.append(episode_grads)
        return scores,rewards,grads

    def training_protocol(self,number_of_episodes,nn,loss_fn,optimizer,gradient_update_period=1,factor=0.5)->Tuple[List,List,List]:
        """
        """
        scores=[]
        rewards=[]
        grads=[]
        method='roulette_prob'
        for episode in range(1,number_of_episodes+1):
            score,episode_rewards,episode_grads=self.run_episode(nn,loss_fn,method)
            episode_rewards=self.discount_reward_episode(episode_rewards,factor)
            episode_grads=np.asarray(episode_grads)
            scores.append(score)
            rewards.append(episode_rewards)
            grads.append(episode_grads)

            

            if episode%gradient_update_period==0:
                batch_rewards=rewards[-gradient_update_period:]
                batch_grads=grads[-gradient_update_period:]
                batch_rewards=self.normalize_matrix(batch_rewards)
                batch_grads=[self.apply_reward_weight_in_grads(episode_rewards,episode_grads)  for episode_rewards,episode_grads in zip(batch_rewards,batch_grads)] 
                batch_grads=self.average_grads_from_episodes(nn,batch_grads)
                self.apply_police_gradient(optimizer,batch_grads,nn) 
                if not self.hold_results: 
                    del batch_rewards
                    del batch_grads
                    del rewards
                    del grads
                    rewards=[]
                    grads=[]
                else:
                    self.rewads.extend(rewards)
                    self.grads.extend(grads)
                    self.scores.extend(scores)
           


        return scores,rewards,grads

if __name__=="__main__":
    PolicyGradient(1,3,4,5)
    print('Tested')
