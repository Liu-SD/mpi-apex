from mpi4py import MPI
from arguments import argparser
from learner import learner
from actor import actor
from eval import evaluator
from replay import replay
import utils
import pickle
import torch

def set_context(n_actors):
    utils.comm = MPI.COMM_WORLD
    assert utils.comm.Get_size() == n_actors + 3
    utils.RANK_ACTORS = [i for i in range(n_actors)]
    utils.RANK_REPLAY = n_actors
    utils.RANK_LEARNER = n_actors+1
    utils.RANK_EVALUATOR = n_actors+2

if __name__ == '__main__':
    torch.set_num_threads(1)
    args = argparser()
    set_context(args.num_actors)
    rank = utils.comm.Get_rank()
    if rank == utils.RANK_LEARNER:
        learner(args)
    elif rank == utils.RANK_REPLAY:
        replay(args)
    elif rank == utils.RANK_EVALUATOR:
        evaluator(args)
    else:
        actor_id = rank
        actor(args, actor_id)
