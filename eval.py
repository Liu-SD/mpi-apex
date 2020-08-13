import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils
from utils import global_dict
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from mpi4py import MPI
from datetime import datetime
import pickle
import os

def evaluator(args):
    comm = global_dict['comm_world']
    writer = SummaryWriter(log_dir=os.path.join(args['log_dir'], 'eval'))

    args['clip_rewards'] = False
    args['episode_life'] = False
    env = make_atari(args['env'])
    env = wrap_atari_dqn(env, args)

    seed = args['seed'] - 1
    utils.set_global_seeds(seed, use_torch=True)
    env.seed(seed)

    torch.set_num_threads(1)
    model = DuelingDQN(env, args)

    recv_param_buf = bytearray(100*1024*1024)
    comm.Send(b'', dest=global_dict['rank_learner'])
    comm.Recv(buf=recv_param_buf, source=global_dict['rank_learner'])
    param = pickle.loads(recv_param_buf)
    model.load_state_dict(param)

    episode_reward, episode_length, episode_idx = 0, 0, 0
    state = env.reset()
    tb_dict = {k: [] for k in ['episode_reward', 'episode_length']}
    while True:
        action, _ = model.act(torch.FloatTensor(np.array(state)), 0.)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done or episode_length == args['max_episode_length']:
            state = env.reset()
            tb_dict["episode_reward"].append(episode_reward)
            tb_dict["episode_length"].append(episode_length)
            episode_reward = 0
            episode_length = 0
            episode_idx += 1
            comm.Send(b'', dest=global_dict['rank_learner'])
            comm.Recv(buf=recv_param_buf, source=global_dict['rank_learner'])
            param = pickle.loads(recv_param_buf)
            model.load_state_dict(param)

            if (episode_idx * args['num_envs_per_worker']) % args['tb_interval'] == 0:
                writer.add_scalar('evaluator/1_episode_reward_mean', np.mean(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/2_episode_reward_max', np.max(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/3_episode_reward_min', np.min(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/4_episode_reward_std', np.std(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/5_episode_length_mean', np.mean(tb_dict['episode_length']), episode_idx)
                tb_dict['episode_reward'].clear()
                tb_dict['episode_length'].clear()
