import os

def save_score_results_for_latter_comparison(factor,gradient_update_period,lr,iterations):
        os.rename('scores/scores.txt',f'scores/scores_fator_{factor}_periodo_gradiente_{gradient_update_period}_lr_{lr}_iteratins_{iterations}.txt')

def save_model(model,factor,gradient_update_period,lr,iterations):
     title=f'models/model_fator_{factor}_periodo_gradiente_{gradient_update_period}_lr_{lr}_iterations_{iterations}.h5'
     model.save(title)