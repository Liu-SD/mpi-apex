import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from model import DuelingDQN
from memory import BatchStorage
from wrapper import make_atari, wrap_atari_dqn
import utils
from mpi4py import MPI
from mpi4py.MPI import Request
import os

def actor(args, actor_id, comm):
    logger = utils.get_logger(f'actor{actor_id}')
    logger.info(f'actor: rank={comm.Get_rank()}, actor_id={actor_id}, pid={os.getpid()}')
    writer = SummaryWriter(comment="-{}-actor{}".format(args.env, actor_id))

    num_envs = args.num_envs_per_worker
    envs = [wrap_atari_dqn(make_atari(args.env), args) for _ in range(num_envs)]

    if args.seed is not None:
        seeds = args.seed + actor_id * num_envs + np.arange(num_envs)
        utils.set_global_seeds(seeds[0], use_torch=True)
        for seed, env in zip(seeds, envs):
            env.seed(int(seed))

    model = DuelingDQN(envs[0], args)
    model = torch.jit.trace(model, torch.zeros((1,4,84,84)))
    param_recv_buffer = bytearray(100*1024*1024)
    _actor_id = np.arange(num_envs) + actor_id * num_envs
    n_actors = args.num_actors * num_envs
    epsilons = args.eps_base ** (1 + _actor_id / (n_actors - 1) * args.eps_alpha)
    storages = [BatchStorage(args.n_steps, args.gamma) for _ in range(num_envs)]

    recv_param_request = None
    send_batch_request_queue = []
    send_batch_request_maxsize = args.max_outstanding

    actor_idx = 0
    tb_idx = 0
    episode_rewards = np.array([0] * num_envs)
    episode_lengths = np.array([0] * num_envs)
    states = np.array([env.reset() for env in envs])
    tb_dict = {key: [] for key in ['episode_reward', 'episode_length']}
    step_t = time.time()
    ref_t = 0
    sim_t = 0

    episode_lengths_running_mean = np.array([1] * num_envs)
    running_mean_alpha = 0.99
    def make_episilons(): # cosine schedule
        return epsilons
        scale = np.cos(np.pi + np.pi * episode_lengths / episode_lengths_running_mean) / 2 + 0.5
        scale[episode_lengths > episode_lengths_running_mean] = 1
        return epsilons * scale

    while True:
        if recv_param_request is not None:
            ready, param = recv_param_request.test()
            if ready:
                model.load_state_dict(param)
                recv_param_request = None
        if actor_idx * num_envs * n_actors <= args.initial_exploration_samples: # initial random exploration
            random_idx = np.arange(num_envs)
        else:
            random_idx, = np.where(np.random.random(num_envs) <= make_episilons())
        _t = time.time()
        with torch.no_grad():
            states_tensor = torch.tensor(states, dtype=torch.float32)
            q_values = model(states_tensor).detach().numpy()
        ref_t += time.time() - _t
        actions = np.argmax(q_values, 1)
        actions[random_idx] = np.random.choice(envs[0].action_space.n, len(random_idx))

        for i, (state, q_value, action, env, storage) in enumerate(zip(states, q_values, actions, envs, storages)):
            _t = time.time()
            next_state, reward, done, _ = env.step(action)
            sim_t += time.time() - _t
            storage.add(np.array(state), reward, action, done, q_value, _t, episode_lengths[i])
            states[i] = next_state
            episode_rewards[i] += reward
            episode_lengths[i] += 1
            if done or episode_lengths[i] == args.max_episode_length:
                states[i] = env.reset()
                episode_lengths_running_mean[i] = \
                    episode_lengths_running_mean[i] * running_mean_alpha + episode_lengths[i] * (1 - running_mean_alpha)

                tb_idx += 1
                tb_dict["episode_reward"].append(episode_rewards[i])
                tb_dict["episode_length"].append(episode_lengths[i])
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                if tb_idx % args.tb_interval == 0:
                    writer.add_scalar('actor/episode_reward_mean', np.mean(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_reward_max', np.max(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_reward_min', np.min(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_reward_std', np.std(tb_dict['episode_reward']), tb_idx)
                    writer.add_scalar('actor/episode_length_mean', np.mean(tb_dict['episode_length']), tb_idx)
                    tb_dict['episode_reward'].clear()
                    tb_dict['episode_length'].clear()
                    writer.add_scalar('actor/step_time', (time.time() - step_t) / np.sum(episode_lengths), tb_idx)
                    writer.add_scalar('actor/step_inference_time', ref_t / np.sum(episode_lengths), tb_idx)
                    writer.add_scalar('actor/step_simulation_time', sim_t / np.sum(episode_lengths), tb_idx)
                    ref_t = 0
                    sim_t = 0
                    step_t = time.time()

        actor_idx += 1

        if actor_idx % args.update_interval == 0 and recv_param_request is None:
            comm.isend('', dest=utils.RANK_LEARNER)
            recv_param_request = comm.irecv(buf=param_recv_buffer, source=utils.RANK_LEARNER)

        if sum(len(storage) for storage in storages) >= args.send_interval * num_envs:
            batch = []
            prios = []
            for storage in storages:
                _batch, _prios = storage.make_batch()
                batch.append(_batch)
                prios.append(_prios)
                storage.reset()
            batch = [np.concatenate(v) for v in zip(*batch)]
            prios = np.concatenate(prios)
            if len(send_batch_request_queue) == send_batch_request_maxsize:
                index, _ = Request.waitany(send_batch_request_queue)
                send_batch_request_queue[index] = comm.isend((batch, prios), dest=utils.RANK_REPLAY, tag=utils.TAG_RECV_BATCH)
            else:
                send_batch_request_queue.append(comm.isend((batch, prios), dest=utils.RANK_REPLAY, tag=utils.TAG_RECV_BATCH))
