from mpi4py import MPI
from mpi4py.MPI import Request
from torch.utils.tensorboard import SummaryWriter
import torch
import time
import numpy as np
import utils
from utils import global_dict
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
import os
import queue
import threading
import pickle
import sys

prios_sum = 1.0

def recv_batch(queue):
    comm = global_dict['comm_local']
    comm.Send(b'', dest=global_dict['rank_replay'], tag=utils.TAG_SEND_BATCH)
    recv_batch_buffer = bytearray(50*1024*1024)
    global prios_sum
    while True:
        comm.Recv(buf=recv_batch_buffer, source=global_dict['rank_replay'])
        batch, prios_sum = pickle.loads(recv_batch_buffer)
        queue.put(batch)

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

def send_prios(queue, n_requests=1):
    comm = global_dict['comm_local']
    send_prios_requests = []
    while True:
        prios = queue.get()
        data = pickle.dumps(prios)
        if len(send_prios_requests) < n_requests:
            send_prios_requests.append(comm.Isend(data, dest=global_dict['rank_replay'], tag=utils.TAG_RECV_PRIOS))
        else:
            index = Request.Waitany(send_prios_requests)
            send_prios_requests[index] = comm.Isend(data, dest=global_dict['rank_replay'], tag=utils.TAG_RECV_PRIOS)

def send_param(queue):
    comm = global_dict['comm_local']
    send_param_ranks = global_dict['rank_actors'] + [global_dict['rank_evaluator']]
    bufs = [bytearray(1) for _ in send_param_ranks]
    send_param_requests = [comm.Irecv(buf=bufs[i], source=rank) for i, rank in enumerate(send_param_ranks)]
    while True:
        param = queue.get()
        current_send_param_idxes = [i for i, req in enumerate(send_param_requests) if req.Test()]
        if len(current_send_param_idxes) == 0:
            continue
        for k, v in param.items():
            param[k] = v.cpu()
        data = pickle.dumps(param)
        for i in current_send_param_idxes:
            comm.Send(data, dest=send_param_ranks[i])
            send_param_requests[i] = comm.Irecv(buf=bufs[i], source=send_param_ranks[i])

def learner(args):
    comm = global_dict['comm_local']
    comm_cross = global_dict['comm_cross']
    size_cross = comm_cross.Get_size()
    env = wrap_atari_dqn(make_atari(args.env), args)
    # utils.set_global_seeds(args.seed, use_torch=True)

    device = args.device
    model = DuelingDQN(env, args).to(device)
    if os.path.exists('model.pth'):
        # model.load_state_dict(torch.load('model.pth'))
        pass

    tgt_model = DuelingDQN(env, args).to(device)
    tgt_model.load_state_dict(model.state_dict())
    del env

    writer = SummaryWriter(comment="-{}-{}-learner".format(args.env, global_dict['unit_idx']))
    # optimizer = torch.optim.SGD(model.parameters(), 3e-2)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.95, eps=1.5e-7, centered=True)

    batch_queue = queue.Queue(maxsize=3)
    tensor_queue = queue.Queue(maxsize=4)
    prios_queue = queue.Queue(maxsize=4)
    param_queue = queue.Queue(maxsize=3)
    threading.Thread(target=recv_batch, args=(batch_queue,)).start()
    threading.Thread(target=to_tensor, args=(batch_queue, tensor_queue, device)).start()
    threading.Thread(target=send_prios, args=(prios_queue,)).start()
    threading.Thread(target=send_param, args=(param_queue,)).start()

    learn_idx = 0
    ts = time.time()
    tb_dict = {k: [] for k in ['loss', 'grad_norm', 'max_q', 'mean_q', 'min_q', 'batch_queue_size', 'tensor_queue_size', 'prios_queue_size']}
    while True:
        (*batch, idxes) = tensor_queue.get()
        loss, prios, q_values = utils.compute_loss(model, tgt_model, batch, args.n_steps, args.gamma)

        optimizer.zero_grad()
        loss.backward()
        global_prios_sum = np.array(prios_sum)
        comm_cross.Allreduce(MPI.IN_PLACE, global_prios_sum.data)
        global_prios_sum = float(global_prios_sum)
        scale = prios_sum / global_prios_sum
        grad_norm = 0.
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm ** (1. / 2)

            if learn_idx == 1e4:
                state_dict = model.state_dict()
                for k, v in state_dict.items():
                    v = v.cpu().numpy()
                    comm_cross.Bcast(v.data, root=0)
                    state_dict[k] = torch.Tensor(v).to(device)
                model.load_state_dict(state_dict)
            grad = p.grad.cpu().numpy() * scale
            comm_cross.Allreduce(MPI.IN_PLACE, grad.data)
            grad = torch.Tensor(grad).to(device)
            if learn_idx >= 1e4:
                p.grad = grad
        grad_norm = grad_norm  ** (1. / 2)
        optimizer.step()

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
        if learn_idx % args.publish_param_interval == 0:
            param_queue.put(model.state_dict())
        if learn_idx % args.tb_interval == 0:
            bps = args.tb_interval / (time.time() - ts)
            for i, (k, v) in enumerate(tb_dict.items()):
                writer.add_scalar(f'learner/{i+1}_{k}', np.mean(v), learn_idx)
                v.clear()
            writer.add_scalar(f"learner/{i+2}_BPS", bps, learn_idx)
            ts = time.time()
