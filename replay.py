from memory import CustomPrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from mpi4py import MPI
from mpi4py.MPI import Request
import time
import utils
from utils import global_dict
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle
import sys
import numpy as np
import traceback

push_size = 0
sample_size = 0

def worker(args, task_type, buf, lock, buffer, start_sending_batch_condition):
    try:
        comm = global_dict['comm_local']
        global push_size, sample_size
        if task_type == 0: # batch recv
            batch, prios = pickle.loads(buf)
            with lock:
                # t = time.time()
                for i, sample in enumerate(zip(*batch, prios)):
                    buffer.add(*sample)
                # print('recv batch', time.time() - t)
            push_size += i+1
            if len(buffer) - (i+1) < args['threshold_size'] and len(buffer) >= args['threshold_size']:
                with start_sending_batch_condition:
                    start_sending_batch_condition.notify_all()
        elif task_type == 1: # batch send
            if len(buffer) < args['threshold_size']:
                with start_sending_batch_condition:
                    start_sending_batch_condition.wait()
            assert len(buffer) >= args['threshold_size']
            beta = args['beta']
            beta_step = (1 - beta) / 1e7 * args['batch_size']

            send_batch_request = None
            while True:
                # t1 = time.time()
                with lock:
                    # t2 = time.time()
                    beta = min(1, beta + beta_step)
                    batch = buffer.sample(args['batch_size'], beta)
                    prios_sum = buffer._it_sum.sum()
                # t3 = time.time()
                data = pickle.dumps((batch, prios_sum))
                # t4 = time.time()
                if send_batch_request is not None:
                    send_batch_request.wait()
                # t5 = time.time()
                # print(t2-t1, t3-t2, t4-t3, t5-t4)
                # sys.stdout.flush()
                send_batch_request = comm.Isend(data, dest=global_dict['rank_learner'])
                sample_size += args['batch_size']
        elif task_type == 2: # prios recv
            idxes, prios = pickle.loads(buf)
            with lock:
                buffer.update_priorities(idxes, prios)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


def replay(args):
    comm = global_dict['comm_local']
    prev_t = time.time()
    global push_size, sample_size
    writer = SummaryWriter(log_dir=os.path.join(args['log_dir'], f'{global_dict["unit_idx"]}-replay'))
    tb_step = 0

    buffer = CustomPrioritizedReplayBuffer(args['replay_buffer_size'], args['alpha'])
    bufs = [
        [bytearray(50*1024*1024) for _ in range(3)],
        bytearray(1),
        bytearray(10*1024),
    ]
    _n = 0
    requests = [
        comm.Irecv(buf=bufs[0][_n], source=MPI.ANY_SOURCE, tag=utils.TAG_RECV_BATCH),
        comm.Irecv(buf=bufs[1], source=global_dict['rank_learner'], tag=utils.TAG_SEND_BATCH),
        comm.Irecv(buf=bufs[2], source=global_dict['rank_learner'], tag=utils.TAG_RECV_PRIOS),
        ]

    lock = threading.Lock()
    thread_pool_executor = ThreadPoolExecutor(max_workers=10)
    start_sending_batch_condition = threading.Condition()
    pending_count_record = []

    while True:
        index = Request.Waitany(requests)
        thread_pool_executor.submit(worker, args, index, bufs[index] if index!=0 else bufs[0][_n], lock, buffer, start_sending_batch_condition)
        if index == 0: # batch recv
            _n = (_n + 1) % 3
            requests[0]=comm.Irecv(buf=bufs[0][_n], source=MPI.ANY_SOURCE, tag=utils.TAG_RECV_BATCH)
        elif index == 1: # batch send
            requests[1]=comm.Irecv(buf=bufs[1], source=global_dict['rank_learner'], tag=utils.TAG_SEND_BATCH)
        elif index == 2: # prios recv
            requests[2] = comm.Irecv(buf=bufs[2], source=global_dict['rank_learner'], tag=utils.TAG_RECV_PRIOS)
        pending_count_record.append(thread_pool_executor._work_queue.qsize())

        delta_t = time.time() - prev_t
        if delta_t > 60:
            tb_step += 1
            writer.add_scalar('replay/1_push_per_second', push_size / delta_t, tb_step)
            writer.add_scalar('replay/2_sample_per_second', sample_size / delta_t, tb_step)
            writer.add_scalar('replay/3_buffer_size',len(buffer), tb_step)
            writer.add_scalar('replay/4_pending_count', np.mean(pending_count_record), tb_step)
            writer.add_scalar('replay/5_priorities_sum', buffer._it_sum.sum(), tb_step)
            sample_size = 0
            push_size = 0
            prev_t = time.time()
            pending_count_record.clear()
