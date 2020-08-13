import os

import torch
import numpy as np

import utils
from model import DuelingDQN
from wrapper import make_atari, wrap_atari_dqn
from arguments import argparser
import cv2
import os

def main():
    args = argparser()

    args['clip_rewards'] = False
    args['episode_life'] = False
    env = make_atari(args['env'])
    env = wrap_atari_dqn(env, args)

    # seed = args['seed'] + 1122
    # utils.set_global_seeds(seed, use_torch=True)
    # env.seed(seed)

    model = DuelingDQN(env, args)
    model.load_state_dict(torch.load('model.pth'))

    episode_reward, episode_length = 0, 0
    state = env.reset()
    if not os.path.exists('plays'):
        os.mkdir('plays')
    imgs = []
    while True:
        img = env.render(mode='rgb_array')
        imgs.append(img)
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
        value, action = model(state).max(1)
        value = value[0]
        action = action[0]
        next_state, reward, done, _ = env.step(int(action))

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            state = env.reset()
            print("Episode Length / Reward: {} / {}, making video...".format(episode_length, episode_reward), end="")
            video = cv2.VideoWriter(f'plays/{args['env']}-{episode_reward}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (160, 210))
            for img in imgs:
                video.write(img)
            video.release()
            print("done")
            episode_reward = 0
            episode_length = 0
            imgs = []


if __name__ == '__main__':
    # torch.set_num_threads(1)
    main()
