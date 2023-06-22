from PolicyGradient.Base.BasePolicyGradient import BasePolicyGradient
from PolicyGradient.Base.BasePolicyGradient import *
import multiprocessing
import copy

class PolicyGradient(BasePolicyGradient):
    def __init__(self,n_cpus,*args,**kwargs) -> None:
        """
        """
        super().__init__(*args,**kwargs)
        self.n_cpus=n_cpus
        self.envs=[copy.deepcopy(self.env) for i in range(self.n_cpus)]
        
            
    def run_episodes(self,number_of_episodes,method)->Tuple[List,List,List]:
        """
        """
        
        with multiprocessing.Pool(self.n_cpus) as pool:
            results=[pool.apply(self.run_episode_paralel,args=(method,self.envs[i%self.n_cpus])) for i in range(number_of_episodes)]

        scores,rewards,grads=[],[],[]
        for (ep_scores,ep_rewards,ep_grads) in results:
            scores.append(ep_scores)
            rewards.append(ep_rewards)
            grads.append(ep_grads)
        return scores,rewards,grads


    def training_protocol(self,number_of_episodes,optimizer,factor=0.5,threshold=250)->Tuple[List,List,List]:
        """
        """
        scores=[]
        rewards=[]
        grads=[]
        method='roulette_prob'
        changed=False

        updates=number_of_episodes//self.gradient_update_period+int(bool(number_of_episodes%self.gradient_update_period))
        for policy_update in range(1,updates+1):
            episodes_scores,batch_rewards,batch_grads=self.run_episodes(self.gradient_update_period,method=method)
            batch_rewards=[self.discount_reward_episode(episode_rewards,factor) for episode_rewards in batch_rewards ]
            batch_grads=[np.asarray(episode_grads)for episode_grads in batch_grads]
            scores.extend(episodes_scores)
            rewards.extend(batch_rewards)
            grads.extend(batch_grads)
            

            batch_rewards=self.normalize_matrix(batch_rewards)
            batch_grads=[self.apply_reward_weight_in_grads(episode_rewards,episode_grads)  for episode_rewards,episode_grads in zip(batch_rewards,batch_grads)] 
            batch_grads=self.average_grads_from_episodes(self.nn,batch_grads)
            self.apply_police_gradient(optimizer,batch_grads,self.nn)  
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

            if np.median(scores[-self.gradient_update_period:])>threshold*.5 and not changed:                
                lr=keras.backend.get_value(optimizer.learning_rate)
                keras.backend.set_value(optimizer.learning_rate, lr*0.1)
                self.nn.save(f'model_threshold_50%_{threshold}.h5')
                print(f'learning rate changed from {lr} to {lr*0.1}')
                changed=True

            if np.median(scores[-self.gradient_update_period:])>threshold:
                    self.nn.save(f'model_threshold_{threshold}.h5')
                    break

        return scores,rewards,grads
    
    def __str__(self) -> str:
        return 'ParalelizedPolicyGradient'
