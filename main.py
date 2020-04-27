from mpi4py import MPI
from arguments import argparser
from learner import learner
from actor import actor
from eval import evaluator
from replay import replay
import utils
import pickle

def set_rank(n_actors):
    utils.RANK_ACTORS = [i for i in range(n_actors)]
    utils.RANK_REPLAY = n_actors
    utils.RANK_LEARNER = n_actors+1
    utils.RANK_EVALUATOR = n_actors+2

if __name__ == '__main__':
    torch.set_num_threads(1)
    args = argparser()
    set_rank(args.num_actors)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == utils.RANK_LEARNER:
        learner(args, comm)
    elif rank == utils.RANK_REPLAY:
        replay(args, comm)
    elif rank == utils.RANK_EVALUATOR:
        evaluator(args, comm)
    else:
        actor_id = rank
        actor(args, actor_id, comm)
