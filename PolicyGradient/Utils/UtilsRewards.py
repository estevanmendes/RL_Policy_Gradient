import numpy as np
from itertools import chain


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
