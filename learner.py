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

def recv_batch(queue, comm, n_requests=3):
    recv_batch_buffer = [bytearray(50 * 1024 * 1024) for _ in range(n_requests)]
    for _ in range(n_requests):
        comm.isend('', dest=utils.RANK_REPLAY, tag=utils.TAG_SEND_BATCH)
    recv_batch_requests = [comm.irecv(buf=recv_batch_buffer[i], source=utils.RANK_REPLAY) for i in range(n_requests)]
    while True:
        index, msg = Request.waitany(recv_batch_requests)
        comm.isend('', dest=utils.RANK_REPLAY, tag=utils.TAG_SEND_BATCH)
        recv_batch_requests[index] = comm.irecv(buf=recv_batch_buffer[index], source=utils.RANK_REPLAY)
        queue.put(msg)

def to_tensor(in_queue, out_queue, device):
    while True:
        _t_start = time.time()
        msg = in_queue.get()
        _t_req = time.time()

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
        _t_device = time.time()

        out_queue.put(batch)

        _req = _t_req - _t_start
        _device = _t_device - _t_req
        _total = _t_device - _t_start
        # logger.info(f'{_req/_total:.2f}, {_device/_total:.2f}')

def send_prios(queue, comm, logger, n_requests=1):
    send_prios_requests = []
    while True:
        prios = queue.get()
        if len(send_prios_requests) < n_requests:
            send_prios_requests.append(comm.isend(prios, dest=utils.RANK_REPLAY, tag=utils.TAG_RECV_PRIOS))
        else:
            index, _ = Request.waitany(send_prios_requests)
            send_prios_requests[index] = comm.isend(prios, dest=utils.RANK_REPLAY, tag=utils.TAG_RECV_PRIOS)

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

    batch_queue = queue.Queue(maxsize=16)
    tensor_queue = queue.Queue(maxsize=16)
    prios_queue = queue.Queue(maxsize=16)
    threading.Thread(target=recv_batch, args=(batch_queue, comm)).start()
    threading.Thread(target=to_tensor, args=(batch_queue, tensor_queue, device)).start()
    threading.Thread(target=send_prios, args=(prios_queue, comm, logger)).start()

    send_param_ranks = utils.RANK_ACTORS + [utils.RANK_EVALUATOR]
    send_param_requests = [comm.irecv(source=rank) for rank in send_param_ranks]

    learn_idx = 0
    ts = time.time()
    tb_dict = {k: [] for k in ['loss', 'grad_norm', 'max_q', 'mean_q', 'min_q']}
    while True:
        # test send_param_requests
        # there's bug in Request.testsome, so we should using for loop to test them.
        _t_start = time.time()
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
            for rank in send_param_indexes:
                comm.send(param, dest=rank)
        _t_param_send = time.time()

        (*batch, idxes) = tensor_queue.get()
        _t_batch_recv = time.time()

        loss, prios, q_values = utils.compute_loss(model, tgt_model, batch, args.n_steps, args.gamma)
        grad_norm = utils.update_parameters(loss, model, optimizer, args.max_norm)
        _t_train = time.time()
        prios_queue.put((idxes, prios))
        _t_send_prios = time.time()

        _recv_batch = _t_batch_recv - _t_param_send
        _train = _t_train - _t_batch_recv
        _send_prios = _t_send_prios - _t_train
        _total = _t_send_prios - _t_start
        logger.info(f'send param: {len(send_param_indexes)} {_t_param_send - _t_start:.4f} \
recv batch: {_recv_batch:.4f} {int(_recv_batch/_total*100)}% \
train: {_train:.4f} {int(_train/_total*100)}% \
send prios: {_send_prios:.4f} {int(_send_prios/_total*100)}% \
total: {_total:.4f}')

        learn_idx += 1

        tb_dict["loss"].append(float(loss))
        tb_dict["grad_norm"].append(float(grad_norm))
        tb_dict["max_q"].append(float(torch.max(q_values)))
        tb_dict["mean_q"].append(float(torch.mean(q_values)))
        tb_dict["min_q"].append(float(torch.min(q_values)))

        if args.soft_target_update:
            tau = args.tau
            for p_tgt, p in zip(tgt_model.parameters(), model.parameters()):
                p_tgt.data *= 1-tau
                p_tgt.data += tau * p
        elif learn_idx % args.target_update_interval == 0:
            logger.info("Updating Target Network..")
            tgt_model.load_state_dict(model.state_dict())
        if learn_idx % args.save_interval == 0:
            logger.info("Saving Model..")
            torch.save(model.state_dict(), "model.pth")
        if learn_idx % args.tb_interval == 0:
            bps = args.tb_interval / (time.time() - ts)
            logger.info("Step: {:8} / BPS: {:.2f}".format(learn_idx, bps))
            writer.add_scalar("learner/BPS", bps, learn_idx)
            for k, v in tb_dict.items():
                writer.add_scalar(f'learner/{k}', np.mean(v), learn_idx)
                v.clear()
            ts = time.time()
