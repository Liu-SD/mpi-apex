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

    args.clip_rewards = False
    args.episode_life=False
    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    # seed = args.seed + 1122
    # utils.set_global_seeds(seed, use_torch=True)
    # env.seed(seed)

    model = DuelingDQN(env, args)
    model.load_state_dict(torch.load('model.pth', map_location='cpu'))

    episode_reward, episode_length = 0, 0
    state = env.reset()
    if not os.path.exists('plays'):
        os.mkdir('plays')
    video = cv2.VideoWriter('plays/tmp.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (160, 210))
    while True:
        img = env.render(mode='rgb_array')
        model.zero_grad()
        state = torch.tensor(state[np.newaxis, :], dtype=torch.float32, requires_grad=True)
        value, action = model(state).max(1)
        value = value[0]
        action = action[0]
        value.backward()
        img_gradient = np.abs(state.grad.numpy())
        img_gradient = np.sum(img_gradient, axis=(0,1))
        img_gradient = (img_gradient - np.min(img_gradient)) / (np.max(img_gradient) - np.min(img_gradient))
        img_gradient = img_gradient.transpose()
        img_gradient = cv2.resize(img_gradient, (160, 210))[...,np.newaxis]
        img_gradient = img_gradient * 255
        masked_img = (img + img_gradient).astype(np.uint8)
        masked_img = np.clip(masked_img, 0, 255)
        video.write(masked_img)
        next_state, reward, done, _ = env.step(int(action))

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done:
            state = env.reset()
            print("Episode Length / Reward: {} / {}".format(episode_length, episode_reward))
            video.release()
            os.rename('plays/tmp.avi', f'plays/{args.env}-{episode_reward}.avi')
            video = cv2.VideoWriter('plays/tmp.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (160, 210))
            episode_reward = 0
            episode_length = 0


if __name__ == '__main__':
    main()
