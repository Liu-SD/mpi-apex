import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from model import DuelingDQN
from memory import BatchStorage
from wrapper import make_atari, wrap_atari_dqn
import utils
from utils import global_dict
from mpi4py import MPI
from mpi4py.MPI import Request
import os
import pickle
import queue
import threading
import sys

def actor(args, actor_id):
    comm = global_dict['comm_local']
    writer = SummaryWriter(log_dir=os.path.join(args['log_dir'], f'{global_dict["unit_idx"]}-actor{actor_id}'))

    num_envs = args['num_envs_per_worker']
    envs = [wrap_atari_dqn(make_atari(args['env']), args) for _ in range(num_envs)]

    if args['seed'] is not None:
        seeds = args['seed'] + actor_id * num_envs + np.arange(num_envs)
        utils.set_global_seeds(seeds[0], use_torch=True)
        for seed, env in zip(seeds, envs):
            env.seed(int(seed))

    model = DuelingDQN(envs[0], args)
    model = torch.jit.trace(model, torch.zeros((1,4,84,84)))
    _actor_id = (np.arange(num_envs) + actor_id * num_envs) * args['num_units'] + global_dict['unit_idx']
    n_actors = args['num_actors'] * num_envs * args['num_units']
    epsilons = args['eps_base'] ** (1 + _actor_id / (n_actors - 1) * args['eps_alpha'])
    storages = [BatchStorage(args['n_steps'], args['gamma']) for _ in range(num_envs)]

    recv_param_buf = bytearray(100*1024*1024)
    recv_param_request = None
    send_batch_request = None

    actor_idx = 0
    tb_idx = 0
    episode_rewards = np.array([0] * num_envs)
    episode_lengths = np.array([0] * num_envs)
    states = np.array([env.reset() for env in envs])
    tb_dict = {key: [] for key in ['episode_reward', 'episode_length', 'kept_sample_percentage']}
    step_t = time.time()
    inf_t = 0
    sim_t = 0

    def make_episilons():
        return epsilons

    while True:
        if recv_param_request and recv_param_request.Test():
            param = pickle.loads(recv_param_buf)
            model.load_state_dict(param)
            recv_param_request = None
        if actor_idx * num_envs * n_actors <= args['initial_exploration_samples']: # initial random exploration
            random_idx = np.arange(num_envs)
        else:
            random_idx, = np.where(np.random.random(num_envs) <= make_episilons())
        _t = time.time()
        with torch.no_grad():
            states_tensor = torch.tensor(states, dtype=torch.float32)
            q_values = model(states_tensor).detach().numpy()
        inf_t += time.time() - _t
        actions = np.argmax(q_values, 1)
        actions[random_idx] = np.random.choice(envs[0].action_space.n, len(random_idx))

        for i, (state, q_value, action, env, storage) in enumerate(zip(states, q_values, actions, envs, storages)):
            _t = time.time()
            next_state, reward, done, _ = env.step(action)
            try:
                real_done = env.was_real_done
            except:
                real_done = done
            sim_t += time.time() - _t
            storage.add(np.array(state), reward, action, done, real_done, q_value, _t, episode_lengths[i])
            states[i] = next_state
            episode_rewards[i] += reward
            episode_lengths[i] += 1
            if done or episode_lengths[i] == args['max_episode_length']:
                states[i] = env.reset()
            if real_done or episode_lengths[i] == args['max_episode_length']:
                tb_idx += 1
                tb_dict["episode_reward"].append(episode_rewards[i])
                tb_dict["episode_length"].append(episode_lengths[i])
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                if tb_idx % args['tb_interval'] == 0:
                    writer.add_scalar('actor/1_episode_reward_mean', np.mean(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/2_episode_reward_max', np.max(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/3_episode_reward_min', np.min(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/4_episode_reward_std', np.std(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/5_episode_length_mean', np.mean(tb_dict['episode_length']), tb_idx)
                    tb_dict['episode_reward'].clear()
                    tb_dict['episode_length'].clear()
                    writer.add_scalar('actor/6_step_time', (time.time() - step_t) / np.sum(episode_lengths), tb_idx)
                    writer.add_scalar('actor/7_step_inference_time', inf_t / np.sum(episode_lengths), tb_idx)
                    writer.add_scalar('actor/8_step_simulation_time', sim_t / np.sum(episode_lengths), tb_idx)
                    writer.add_scalar('actor/9_kept_sample_percentage', np.mean(tb_dict['kept_sample_percentage']), tb_idx)
                    inf_t = 0
                    sim_t = 0
                    step_t = time.time()
                    tb_dict['kept_sample_percentage'].clear()

        actor_idx += 1

        if actor_idx % args['update_interval'] == 0:
            if recv_param_request is not None:
                print(f"actor {global_dict['unit_idx']}-{actor_id}: last recv param request is not complete!")
                sys.stdout.flush()
            else:
                comm.Send(b'', dest=global_dict['rank_learner'])
                recv_param_request = comm.Irecv(buf=recv_param_buf, source=global_dict['rank_learner'])

        if sum(len(storage) for storage in storages) >= args['send_interval'] * num_envs:
            batch = []
            prios = []
            for storage in storages:
                _batch, _prios = storage.make_batch()
                batch.append(_batch)
                prios.append(_prios)
                storage.reset()
            batch = [np.concatenate(v) for v in zip(*batch)]
            prios = np.concatenate(prios)
            threshold = args['sample_filter_threshold']
            prios_mask = prios > np.max(prios) * threshold
            tb_dict['kept_sample_percentage'].append(np.sum(prios_mask) / len(prios_mask))
            prios = prios[prios_mask]
            batch = [i[prios_mask] for i in batch]
            data = pickle.dumps((batch, prios))
            if send_batch_request is not None:
                send_batch_request.wait()
            send_batch_request = comm.Isend(data, dest=global_dict['rank_replay'], tag=utils.TAG_RECV_BATCH)
