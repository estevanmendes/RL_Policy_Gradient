import numpy as np
from tensorflow import GradientTape,reduce_mean,Tensor
from abc import abstractclassmethod
import keras
from PolicyGradient.Utils import UtilsRewards
from PolicyGradient.Utils import UtilsGrads
from PolicyGradient.Utils import UtilsMetrics
from PolicyGradient.Utils import UtilsSaving

from typing import Any, Tuple,List


class BasePolicyGradient():
    """
    This class is the base for buiding the Policy Gradient and the PolicyGrandientParalel classes.
    It has several static methods for evaluate the reward of each action, apply the gradient, average de gradients, and so on. 
    Also has methods for run a episode with or without paralelization, and to evaluate the loss and the gradient of each action on the enviroment.

    ..

    Attributes
    ----------
    env: 
        enviroment from gym that will be used to trainning the model.
    state_size: int,tuple
        the observation shape. 
    min_score_aceptable: int
        the score in which the episode is ended, before it is done.
    nn: 
        A neural network from tensor flow that is callable
    loss_fn: keras.losses
        A loss function to evaluate the distance between the model outcome and the target.
    gradient_update_period:
        The number of episode that should be waited before applying the gradients
    hold_results: bool, optional
        If one wants the grads,scores and rewards recorded in the memory for latter access in the attributes.
    metrics_display_period: int,optional
        It controls how many episodes must go by before printing the total number of episodes in the training protocol.
    threshold: int, optional
        The valeu where the algorithm it will stop the trainning protocol, since the median score was reached by the model. IT works as an checkpoint, saving the model on 50% of the goal as well. 
    threshold_window: int, optional
        The window that will be used for calculate median and check if the score threshold was reached. By deafult it is the gradient_update_period.


    Methods
    -------
    action_and_grads()
        It computes the model action given the state recording the steps with tape for the autodiff to compute the gradients.

    run_episode()
        It runs all steps in the enviroment before the episode is done.
        It resets the env, gets the action, takes a step in the env, records the grads,rewards, scores.  

    run_episode_paralel()
        It receaves the model as parameter diferently than the oprdinary method, because, otherwise, the multiprocessing lybrary doesnot work. 
        It runs all steps in the enviroment before the episode is done.
        It resets the env, gets the action, takes a step in the env, records the grads,rewards, scores.  

    run_multiple_episodes()
        It manages the multiple episodes calls.

    training_protocol()
        It runs the number of seted episodes, applies the reward policy, normalizes the rewards, takes the weighted average between the grads the normalized rewards,
        applies the grads using the optimizer.

    discount_rewards_multiple_episodes()
        It applies the discount in future rewards to account it into the previus actions. 
    
    normalize_matrix(),
        It normalizes a matrix, which in this context means the normalization of multiple episodes. 

    apply_reward_weight_in_grads()
         It weight averages the gradients with the rewards of each action

    average_grads_from_episodes()
        It averages the gradients of multiples episodes, for each model variable It is taken the arithmetic mean.
       

    apply_police_gradient()
        It applies the gradients in the neural network. 


    display_score()
        It plots a graph of the score, i.e. the sum of rewards, curve by the episodes. It is applied a moving average to smooth the curve.
    
    save_trainning()
        It saves the model trainned in the models folder, and the scores in the score folder. 


    """


    @staticmethod
    def discount_reward_episode(*args,**kwargs):
        return UtilsRewards.discount_reward_episode(*args,**kwargs)
    
    @staticmethod
    def discount_rewards_multiple_episodes(*args,**kwargs):
        """
        It applies the discount in future rewards to account it into the previus actions. 
        
        Parameters
        ----------
        rewards: list,array
            the serie of rewards of each epsidode
        factor: float
            the factor that multiply the reward of each step
        """
        return UtilsRewards.discount_rewards_multiple_episodes(*args,**kwargs)

    @staticmethod
    def normalize_vector(*args,**kwargs):
        return UtilsRewards.normalize_vector(*args,**kwargs)

    @staticmethod
    def normalize_matrix(*args,**kwargs):
        """
        It normalizes a matrix, which in this context means the normalization of multiple episodes. 
        
        Parameters
        ----------
        matrix: np.array
        """
        return UtilsRewards.normalize_matrix(*args,**kwargs)

    @staticmethod
    def apply_reward_weight_in_grads(*args,**kwargs):
        """
        It weight averages the gradients with the rewards of each action
        
        Parameters
        ----------
        rewards: np.array
            Serie of rewards of an episode
        grads: np.array
            grads of an episode
        """
        return UtilsRewards.apply_reward_weight_in_grads(*args,**kwargs)

    @staticmethod
    def average_grads_from_steps(*args,**kwargs):
        return UtilsGrads.average_grads_from_steps(*args,**kwargs)
    
    @staticmethod
    def average_grads_from_episodes(*args,**kwargs):
        """
        It averages the gradients of multiples episodes, for each model variable It is taken the arithmetic mean.
        
        Parameters
        ----------
        nn: tensorflow.Model
            Callable tensorflow neural network
        grads: list,array
            list of grads of multiple episodes
        """
        return UtilsGrads.average_grads_from_episodes(*args,**kwargs)

    @staticmethod
    def apply_police_gradient(*args,**kwargs):
        """
        It applies the gradients in the neural network. 
        
        Parameters
        ----------
        optimizer: keras.optimizers
            optimizer for apply the gradients.
        nn: tensorflow.Model
            Callable tensorflow neural network
        """
        return UtilsGrads.apply_police_gradient(*args,**kwargs)
    
    @staticmethod
    def display_score(*args,**kwargs):
        """
        It plots a graph of the score, i.e. the sum of rewards, curve by the episodes. It is applied a moving average to smooth the curve.
        
        Parameters
        ----------
        scores:list,array
            The list of float or ints containing the historical serie of scores
        window: int
            the window to apply a moving average. 
        """
        return UtilsMetrics.display_score(*args,**kwargs)



    def __init__(self,env,state_size,min_score_aceptable,nn,loss_fn:keras.losses,gradient_update_period,hold_results=False,metrics_display_period=10*4,threshold=10**6,threshold_window=None) -> None:
        """
        Parameters
        ----------
        env: 
            enviroment from gym that will be used to trainning the model.
        state_size: int,tuple
            the observation shape. 
        min_score_aceptable: int
            the score in which the episode is ended, before it is done.
        nn: 
            A neural network from tensor flow that is callable
        loss_fn: keras.losses
            A loss function to evaluate the distance between the model outcome and the target.
        gradient_update_period:
            The number of episode that should be waited before applying the gradients
        hold_results: bool, optional
            If one wants the grads,scores and rewards recorded in the memory for latter access in the attributes.
        metrics_display_period: int,optional
            It controls how many episodes must go by before printing the total number of episodes in the training protocol.
        threshold: int, optional
            The valeu where the algorithm it will stop the trainning protocol, since the median score was reached by the model. IT works as an checkpoint, saving the model on 50% of the goal as well. 
        threshold_window: int, optional
            The window that will be used for calculate median and check if the score threshold was reached. By deafult it is the gradient_update_period.

        
        """
        self.env=env
        self.min_score_aceptable=min_score_aceptable
        self.state_size=[-1]
        if isinstance(state_size,int):
            self.state_size.append(state_size)
        else:
            self.state_size.extend(state_size)

        self.nn=nn
        self.loss_fn=loss_fn
        self.gradient_update_period=gradient_update_period
        self.threshold=threshold
        if threshold_window is None:
            threshold_window=gradient_update_period

        self.threshold_window=threshold_window
        self.hold_results=hold_results
        self.metrics_display_period=metrics_display_period
        self.grads=[]
        self.rewads=[]
        self.scores=[]
        self._counter=1

    @UtilsMetrics.metrics_method_decorator   
    def run_episode(self,method)->Tuple[List[float],List[float],List[List[Tensor]]]:
        """
        It runs all steps in the enviroment before the episode is done.
        It resets the env, gets the action, takes a step in the env, records the grads,rewards, scores.  

        Parameters
        ----------
        method : str
            The way of chosing the action from the model outcome, which is an array of actions lenth
        """
     
        
        state,info=self.env.reset()
        score=0
        rewards=[]
        grads=[]
        while True:
                action,step_grads=self.action_and_grads(state,self.nn,method=method)
                step_results=self.env.step(action)
                state,reward,done=step_results[0:3]
                score+=reward
                rewards.append(reward)
                grads.append(step_grads)
                if done or score<self.min_score_aceptable:      
                    with open('scores/scores.txt','a+') as f:
                        f.write(f'{score}\n')  

                    break

        return score,rewards,grads
    
    def run_episode_paralel(self,method,env)->Tuple[List[float],List[float],List[List[Tensor]]]:
        """
        It runs all steps in the enviroment before the episode is done.
        It resets the env, gets the action, takes a step in the env, records the grads,rewards, scores.  
        
        
        """
        
        state,info=env.step(None)[0]
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
                    with open('scores/scores.txt','a+') as f:
                        f.write(f'{score}\n')  

                    break
                

        return score,rewards,grads
    



    def action_and_grads(self,state,nn,method='roulette_prob')-> Tuple[int, List]:
        """
        It computes the model action given the state recording the steps with tape for the autodiff to compute the gradients.
        
        Parameters
        ----------
        state: list, array
            The observation of the gym enviroment.
        nn: tensorflow.Model
            A neural network callable
        method: str
            The way of chosing the action from the model outcome, which is an array of actions lenth
            max_prob: It alwayes chooses the max probability action.
            roulette_prob: It randomly chooses tha action based on the probability of each action by the model.

        """
        methods=['max_prob','roulette_prob']
        if method not in methods:
            raise ValueError(f"Invalid method. Expected one of: {methods}")
        
        state=np.array(state).reshape(*self.state_size)
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
        """
        It manages the multiple episodes calls.
        """
        pass
    


    @abstractclassmethod
    def training_protocol(self):
        """
        It runs the number of seted episodes, applies the reward policy, normalizes the rewards, takes the weighted average between the grads the normalized rewards,
        applies the grads using the optimizer.

        """
        pass

    def __str__(self) -> str:
        return 'BasePolicyGradient'

    
    def save_trainning(self,factor='Null',lr='Null'):
        """
        It saves the model trainned in the models folder, and the scores in the score folder. 
        """
        UtilsSaving.save_score_results_for_latter_comparison(factor=factor,gradient_update_period=self.gradient_update_period,lr=lr,iterations=self._counter)
        UtilsSaving.save_model(self.nn,factor=factor,gradient_update_period=self.gradient_update_period,lr=lr,iterations=self._counter)
