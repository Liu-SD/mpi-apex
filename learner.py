from mpi4py import MPI
from mpi4py.MPI import Request
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import numpy as np
import utils
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
import os
import queue
import threading
import pickle
import sys

def recv_batch(queue, comm, n_requests=3):
    recv_batch_buffer = [bytearray(50 * 1024 * 1024) for _ in range(n_requests)]
    for _ in range(n_requests):
        comm.Isend(b'', dest=utils.RANK_REPLAY, tag=utils.TAG_SEND_BATCH)
    recv_batch_requests = [comm.Irecv(buf=recv_batch_buffer[i], source=utils.RANK_REPLAY) for i in range(n_requests)]
    while True:
        index = Request.Waitany(recv_batch_requests)
        comm.Isend(b'', dest=utils.RANK_REPLAY, tag=utils.TAG_SEND_BATCH)
        recv_batch_requests[index] = comm.Irecv(buf=recv_batch_buffer[index], source=utils.RANK_REPLAY)
        msg = pickle.loads(recv_batch_buffer[index])
        queue.put(msg)

def to_tensor(in_queue, out_queue, device):
    torch.set_num_threads(1)
    while True:
        msg = in_queue.get()

        states, actions, rewards, next_states, dones, _, _, weights, idxes = msg
        states = np.array([np.array(state) for state in states])
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = np.array([np.array(state) for state in next_states])
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(weights).to(device)
        batch = [states, actions, rewards, next_states, dones, weights, idxes]

        out_queue.put(batch)

def send_prios(queue, comm, logger, n_requests=1):
    send_prios_requests = []
    while True:
        prios = queue.get()
        data = pickle.dumps(prios)
        if len(send_prios_requests) < n_requests:
            send_prios_requests.append(comm.Isend(data, dest=utils.RANK_REPLAY, tag=utils.TAG_RECV_PRIOS))
        else:
            index = Request.Waitany(send_prios_requests)
            send_prios_requests[index] = comm.Isend(data, dest=utils.RANK_REPLAY, tag=utils.TAG_RECV_PRIOS)

def learner(args, comm):
    logger = utils.get_logger('learner')
    logger.info(f'rank={comm.Get_rank()}, pid={os.getpid()}')
    env = wrap_atari_dqn(make_atari(args.env), args)
    utils.set_global_seeds(args.seed, use_torch=True)

    device = args.device
    model = DuelingDQN(env, args).to(device)
    tgt_model = DuelingDQN(env, args).to(device)
    tgt_model.load_state_dict(model.state_dict())
    del env

    writer = SummaryWriter(comment="-{}-learner".format(args.env))
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.95, eps=1.5e-7, centered=True)

    batch_queue = queue.Queue(maxsize=4)
    tensor_queue = queue.Queue(maxsize=4)
    prios_queue = queue.Queue(maxsize=4)
    threading.Thread(target=recv_batch, args=(batch_queue, comm)).start()
    threading.Thread(target=to_tensor, args=(batch_queue, tensor_queue, device)).start()
    threading.Thread(target=send_prios, args=(prios_queue, comm, logger)).start()

    send_param_ranks = utils.RANK_ACTORS + [utils.RANK_EVALUATOR]
    send_param_requests = [comm.irecv(source=rank) for rank in send_param_ranks]

    learn_idx = 0
    ts = time.time()
    tb_dict = {k: [] for k in ['loss', 'grad_norm', 'max_q', 'mean_q', 'min_q', 'batch_queue_size', 'tensor_queue_size', 'prios_queue_size']}
    while True:
        # test send_param_requests
        # there's bug in Request.testsome, so we should using for loop to test them.
        send_param_indexes = []
        for i, (rank, req) in enumerate(zip(send_param_ranks, send_param_requests)):
            ready, _ = req.test()
            if ready:
                send_param_indexes.append(rank)
                send_param_requests[i] = comm.irecv(source=rank)
        if send_param_indexes:
            param = model.state_dict()
            for k, v in param.items():
                param[k] = v.cpu()
            data = pickle.dumps(param)
            for rank in send_param_indexes:
                comm.Send(data, dest=rank)
        (*batch, idxes) = tensor_queue.get()
        loss, prios, q_values = utils.compute_loss(model, tgt_model, batch, args.n_steps, args.gamma)
        grad_norm = utils.update_parameters(loss, model, optimizer, args.max_norm)
        prios_queue.put((idxes, prios))
        learn_idx += 1
        tb_dict["loss"].append(float(loss))
        tb_dict["grad_norm"].append(float(grad_norm))
        tb_dict["max_q"].append(float(torch.max(q_values)))
        tb_dict["mean_q"].append(float(torch.mean(q_values)))
        tb_dict["min_q"].append(float(torch.min(q_values)))
        tb_dict["batch_queue_size"].append(batch_queue.qsize())
        tb_dict["tensor_queue_size"].append(tensor_queue.qsize())
        tb_dict["prios_queue_size"].append(prios_queue.qsize())

        if args.soft_target_update:
            tau = args.tau
            for p_tgt, p in zip(tgt_model.parameters(), model.parameters()):
                p_tgt.data *= 1-tau
                p_tgt.data += tau * p
        elif learn_idx % args.target_update_interval == 0:
            tgt_model.load_state_dict(model.state_dict())
        if learn_idx % args.save_interval == 0:
            torch.save(model.state_dict(), "model.pth")
        if learn_idx % args.tb_interval == 0:
            bps = args.tb_interval / (time.time() - ts)
            for i, (k, v) in enumerate(tb_dict.items()):
                writer.add_scalar(f'learner/{i+1}_{k}', np.mean(v), learn_idx)
                v.clear()
            writer.add_scalar(f"learner/{i+2}_BPS", bps, learn_idx)
            ts = time.time()
