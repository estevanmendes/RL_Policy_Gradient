## Reinforcement Learning Package

This package provides an easy way of apllying policy gradient into an enviroment. 
It was designed to work with neural networks from TensorFlow, but one may feel invited to extend to Pytorch models as well. 
There is a class in which the algorithm is paralelized into as many cpus as one may want. 

We used a reward discout policy using a bit of linear algebra. THe factor depends on the problem and its score dependency on the previus actions. 

Workflow:
    let the agent explore the enviroment with the neural network
    every N episodes:
        apply the reward discout policy in each episode
        normalize the rewards of the episodes accounting all of them in the normalization
        create a weighted average of the gradient using the normalized rewards
        apply the gradient in the NN using the chosen optimizer
        
    repeat util the number of episodes aimed is reached. 

    

## Results

### Lunar Landing



![lunar example](gifs/LunarLander-v2_score_253.gif)

### Car Racing




## Package Structure. 

Most of the functions inside the Utils package were used as a static mehod inside the BasePolicyGradient Class (which is encopassed on BasePolicyGradine.py file)
The Test package contains very simples test to check if the class are working well. 


PolicyGradient/
|---- Base/
|        |
|        |---__init__.py
|        |---BasePolicyGradine.py
|-----Utils/
|        |
|        |---__init__.py
|        |---UtilsGrads.py
|        |---UtilsRewards.py
|        |---UtilsMetrics.py
|        |---UtilsSaving.py
|-----Tests/
|        |
|        |---__init__.py
|        |---BasePolicyGradient--.py
|        |---PolicyGradientParalel--.py
|        |---PolicyGradientParalel--.py
|
|---PolicyGradient.py
|---PolicyGradientParalel.py