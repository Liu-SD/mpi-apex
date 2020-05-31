from mpi4py import MPI
from arguments import argparser
from learner import learner
from actor import actor
from eval import evaluator
from replay import replay
import utils
from utils import global_dict
import pickle
import torch

def set_context(n_units, n_actors):
    # local rank
    rank_actors = [i for i in range(n_actors)]
    rank_replay = n_actors
    rank_learner = n_actors + 1
    rank_evaluator = n_actors + 2

    comm_world = MPI.COMM_WORLD
    assert comm_world.Get_size() == (n_actors + 3) * n_units
    rank_world = comm_world.Get_rank()

    comm_local = comm_world.Split(rank_world // (n_actors + 3), rank_world)
    rank_local = comm_local.Get_rank()

    group_world = comm_world.Get_group()
    group_cross = group_world.Incl([i * (n_actors + 3) + rank_learner for i in range(n_units)])
    comm_cross = comm_world.Create(group_cross)

    global_dict['rank_actors'] = rank_actors
    global_dict['rank_replay'] = rank_replay
    global_dict['rank_learner'] = rank_learner
    global_dict['rank_evaluator'] = rank_evaluator
    global_dict['unit_idx'] = rank_world // (n_actors + 3)
    global_dict['comm_world'] = comm_world
    global_dict['rank_world'] = rank_world
    global_dict['comm_local'] = comm_local
    global_dict['rank_local'] = rank_local
    global_dict['comm_cross'] = comm_cross # comm_cross is COMM_NULL if I'm not a learner
    if comm_cross != MPI.COMM_NULL:
        global_dict['rank_cross'] = comm_cross.Get_rank()

if __name__ == '__main__':
    torch.set_num_threads(1)
    args = argparser()
    set_context(args.num_units, args.num_actors)
    rank = global_dict['rank_local']
    if rank == global_dict['rank_learner']:
        learner(args)
    elif rank == global_dict['rank_replay']:
        replay(args)
    elif rank == global_dict['rank_evaluator']:
        evaluator(args)
    else:
        actor_id = rank
        actor(args, actor_id)
