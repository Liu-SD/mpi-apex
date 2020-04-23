import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import utils
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from mpi4py import MPI
from datetime import datetime

def evaluator(args, comm):
    logger = utils.get_logger('eval')
    writer = SummaryWriter(comment="-{}-eval".format(args.env))

    args.clip_rewards = False
    args.episode_life = False
    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    seed = args.seed - 1
    utils.set_global_seeds(seed, use_torch=True)
    env.seed(seed)

    model = DuelingDQN(env, args)
    param_recv_buffer = bytearray(100*1024*1024)

    comm.send('', dest=utils.RANK_LEARNER)
    param = comm.recv(buf=param_recv_buffer, source=utils.RANK_LEARNER)
    model.load_state_dict(param)
    logger.info("Received First Parameter!")

    episode_reward, episode_length, episode_idx = 0, 0, 0
    state = env.reset()
    tb_dict = {k: [] for k in ['episode_reward', 'episode_length']}
    while True:
        action, _ = model.act(torch.FloatTensor(np.array(state)), 0.)
        next_state, reward, done, _ = env.step(action)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done or episode_length == args.max_episode_length:
            state = env.reset()
            tb_dict["episode_reward"].append(episode_reward)
            tb_dict["episode_length"].append(episode_length)
            episode_reward = 0
            episode_length = 0
            episode_idx += 1
            comm.send('', dest=utils.RANK_LEARNER)
            param = comm.recv(buf=param_recv_buffer, source=utils.RANK_LEARNER)
            model.load_state_dict(param)
            logger.info(f"{datetime.now()} Updated Parameter..")

            if (episode_idx * args.num_envs_per_worker) % args.tb_interval == 0:
                writer.add_scalar('evaluator/episode_reward_mean', np.mean(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_reward_max', np.max(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_reward_min', np.min(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_reward_std', np.std(tb_dict['episode_reward']), episode_idx)
                writer.add_scalar('evaluator/episode_length_mean', np.mean(tb_dict['episode_length']), episode_idx)
                tb_dict['episode_reward'].clear()
                tb_dict['episode_length'].clear()
