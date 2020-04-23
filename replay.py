from memory import CustomPrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from mpi4py import MPI
from mpi4py.MPI import Request
import time
import utils
import os
from concurrent.futures import ThreadPoolExecutor
import threading

push_size = 0
sample_size = 0
thread_count = 0

def worker(args, task_type, data, lock, buffer, comm, start_sending_batch_condition):
    global push_size, sample_size
    global thread_count
    if task_type == 0: # batch recv
        batch, prios = data
        with lock:
            for i, sample in enumerate(zip(*batch, prios)):
                buffer.add(*sample)
        push_size += i+1
        if len(buffer) - (i+1) < args.threshold_size and len(buffer) >= args.threshold_size:
            with start_sending_batch_condition:
                start_sending_batch_condition.notify_all()
    elif task_type == 1: # batch send
        if len(buffer) < args.threshold_size:
            with start_sending_batch_condition:
                start_sending_batch_condition.wait()
        assert len(buffer) >= args.threshold_size
        with lock:
            batch = buffer.sample(args.batch_size, args.beta)
        comm.send(batch, dest=utils.RANK_LEARNER)
        sample_size += args.batch_size
    elif task_type == 2: # prios recv
        idxes, prios = data
        with lock:
            buffer.update_priorities(idxes, prios)

    thread_count -= 1


def replay(args, comm):
    logger = utils.get_logger('replay')
    logger.info(f'rank={comm.Get_rank()}, pid={os.getpid()}')
    prev_t = time.time()
    global push_size, sample_size
    global thread_count
    writer = SummaryWriter(comment=f"-{args.env}-replay")
    tb_step = 0

    buffer = CustomPrioritizedReplayBuffer(args.replay_buffer_size, args.alpha)
    batch_recv_buff = bytearray(50*1024*1024) # 50M for batch recv buffer space
    requests = [
        comm.irecv(buf=batch_recv_buff, source=MPI.ANY_SOURCE, tag=utils.TAG_RECV_BATCH),
        comm.irecv(source=utils.RANK_LEARNER, tag=utils.TAG_SEND_BATCH),
        comm.irecv(source=utils.RANK_LEARNER, tag=utils.TAG_RECV_PRIOS),
        ]

    lock = threading.Lock()
    thread_pool_executor = ThreadPoolExecutor()
    start_sending_batch_condition = threading.Condition()

    while True:
        _t_start = time.time()
        index, msg = Request.waitany(requests) # always return first meet request, so we should use a queue to make other request replied
        _t_wait = time.time()
        thread_count += 1
        thread_pool_executor.submit(worker, args, index, msg, lock, buffer, comm, start_sending_batch_condition)
        if index == 0: # batch recv
            requests[index]=comm.irecv(buf=batch_recv_buff, source=MPI.ANY_SOURCE, tag=utils.TAG_RECV_BATCH)
        elif index == 1: # batch send
            requests[index]=comm.irecv(source=utils.RANK_LEARNER, tag=utils.TAG_SEND_BATCH)
        elif index == 2: # prios recv
            requests[index] = comm.irecv(source=utils.RANK_LEARNER, tag=utils.TAG_RECV_PRIOS)
        _t_busy = time.time()
        _wait = _t_wait - _t_start
        _busy = _t_busy - _t_wait
        _percentage = _wait / (_busy + _wait)
        request_type_string = ['RECV_BATCH', 'SEND_BATCH', 'RECV_PRIOS']
        logger.info(f'wait time: {_wait:.5f}, busy time: {_busy:.5f}, persentage: {_percentage:.2f}, request: {request_type_string[index]}, current thread counts: {thread_count}')

        delta_t = time.time() - prev_t
        if delta_t > 60:
            tb_step += 1
            writer.add_scalar('replay/push_per_second', push_size / delta_t, tb_step)
            writer.add_scalar('replay/sample_per_second', sample_size / delta_t, tb_step)
            writer.add_scalar('replay/buffer_size',len(buffer), tb_step)
            sample_size = 0
            push_size = 0
            prev_t = time.time()
