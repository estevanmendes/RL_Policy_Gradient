import imageio
import os
from PIL import Image
import PIL.ImageDraw as ImageDraw

def play_episode(env,model,loss_fn,method,save=False,file_to_save_gif=None,threshold=100):
    state=env.reset()
    score=0
    frames=[]
    while True:
            frame=env.render()
            frame=env.render(mode='rgb_array')
            action,step_grads=action_and_grads(state,model,loss_fn,method=method)
            state,reward,done,info=env.step(action)
            score+=reward
            frames.append(_label_with_episode_number(frame,score))
            if done or score<-300:      
                break

    if not file_to_save_gif:
        name = env.unwrapped.spec.id
        file_to_save_gif=name+'_score_'+str(int(score))+'.gif'
    if save and score>threshold:
        imageio.mimwrite(os.path.join('gifs/',file_to_save_gif), frames, duration=len(frames)//60)

    return score

def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Score: {episode_num}', fill=text_color)

    return im