from memory import CustomPrioritizedReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from mpi4py import MPI
from mpi4py.MPI import Request
import time
import utils
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import pickle

push_size = 0
sample_size = 0

def worker(args, task_type, buf, lock, buffer, comm, start_sending_batch_condition):
    global push_size, sample_size
    if task_type == 0: # batch recv
        batch, prios = pickle.loads(buf)
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
        data = pickle.dumps(batch)
        comm.Send(data, dest=utils.RANK_LEARNER)
        sample_size += args.batch_size
    elif task_type == 2: # prios recv
        idxes, prios = pickle.loads(buf)
        with lock:
            buffer.update_priorities(idxes, prios)


def replay(args, comm):
    logger = utils.get_logger('replay')
    logger.info(f'rank={comm.Get_rank()}, pid={os.getpid()}')
    prev_t = time.time()
    global push_size, sample_size
    writer = SummaryWriter(comment=f"-{args.env}-replay")
    tb_step = 0

    buffer = CustomPrioritizedReplayBuffer(args.replay_buffer_size, args.alpha)
    bufs = [
        bytearray(50*1024*1024),
        bytearray(1),
        bytearray(10*1024),
    ]
    requests = [
        comm.Irecv(buf=bufs[0], source=MPI.ANY_SOURCE    , tag=utils.TAG_RECV_BATCH),
        comm.Irecv(buf=bufs[1], source=utils.RANK_LEARNER, tag=utils.TAG_SEND_BATCH),
        comm.Irecv(buf=bufs[2], source=utils.RANK_LEARNER, tag=utils.TAG_RECV_PRIOS),
        ]

    lock = threading.Lock()
    thread_pool_executor = ThreadPoolExecutor()
    start_sending_batch_condition = threading.Condition()

    while True:
        index = Request.Waitany(requests) # always return first meet request, so we should use a queue to make other request replied
        thread_pool_executor.submit(worker, args, index, bufs[index], lock, buffer, comm, start_sending_batch_condition)
        if index == 0: # batch recv
            requests[0]=comm.Irecv(buf=bufs[0], source=MPI.ANY_SOURCE, tag=utils.TAG_RECV_BATCH)
        elif index == 1: # batch send
            requests[1]=comm.Irecv(buf=bufs[1], source=utils.RANK_LEARNER, tag=utils.TAG_SEND_BATCH)
        elif index == 2: # prios recv
            requests[2] = comm.Irecv(buf=bufs[2], source=utils.RANK_LEARNER, tag=utils.TAG_RECV_PRIOS)

        delta_t = time.time() - prev_t
        if delta_t > 60:
            tb_step += 1
            writer.add_scalar('replay/1_push_per_second', push_size / delta_t, tb_step)
            writer.add_scalar('replay/2_sample_per_second', sample_size / delta_t, tb_step)
            writer.add_scalar('replay/3_buffer_size',len(buffer), tb_step)
            writer.add_scalar('replay/4_thread_count', len(thread_pool_executor._threads), tb_step)
            writer.add_scalar('replay/5_pending_count', thread_pool_executor._work_queue.qsize(), tb_step)
            sample_size = 0
            push_size = 0
            prev_t = time.time()
