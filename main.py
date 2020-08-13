from mpi4py import MPI
from learner import learner
from actor import actor
from eval import evaluator
from replay import replay
import utils
from utils import global_dict
import pickle
import torch
import yaml
import sys
import os
from datetime import datetime

def set_context(n_units, n_actors):
    '''
                                  +------------comm_cross------------+-------+
                                  |                                  |       |
    | actor0 actor1 ... replay learner | actor0 actor1 ... replay learner | ... | evaluator |
    | <----------comm_local----------> | <----------comm_local----------> | ... | comm_local|
    | <-----------------------------------comm_world--------------------------------------> |
    '''
    rank_actors = [i for i in range(n_actors)]
    rank_replay = n_actors
    rank_learner = n_actors + 1

    comm_world = MPI.COMM_WORLD
    assert comm_world.Get_size() == (n_actors + 2) * n_units + 1, f'expect {(n_actors + 2) * n_units + 1} procs'
    rank_world = comm_world.Get_rank()

    comm_local = comm_world.Split(rank_world // (n_actors + 2), rank_world)
    rank_local = comm_local.Get_rank()

    group_world = comm_world.Get_group()
    group_cross = group_world.Incl([i * (n_actors + 2) + rank_learner for i in range(n_units)])
    comm_cross = comm_world.Create(group_cross)

    global_dict['rank_actors'] = rank_actors
    global_dict['rank_replay'] = rank_replay
    global_dict['rank_learner'] = rank_learner
    global_dict['rank_evaluator'] = comm_world.Get_size() - 1 # rank of comm_world
    global_dict['unit_idx'] = rank_world // (n_actors + 2)
    global_dict['comm_world'] = comm_world
    global_dict['rank_world'] = rank_world
    global_dict['comm_local'] = comm_local
    global_dict['rank_local'] = rank_local
    global_dict['comm_cross'] = comm_cross # comm_cross is COMM_NULL if I'm not a learner
    if comm_cross != MPI.COMM_NULL:
        global_dict['rank_cross'] = comm_cross.Get_rank()

def parse():
    with open('configs/default.yaml') as f:
        args = yaml.load(f.read())
    if len(sys.argv) > 1:
        custom = sys.argv[1]
        with open(custom) as f:
            custom_args = yaml.load(f.read())
        args.update(custom_args)
    args['cuda'] = args['cuda'] and torch.cuda.is_available()
    log_dir = f'results/{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-{args["env"]}-{args["prefix"]}'
    log_dir = MPI.COMM_WORLD.bcast(log_dir, root=0)
    args['log_dir'] = log_dir
    if MPI.COMM_WORLD.Get_rank() == 0:
        os.makedirs(log_dir)
        with open(log_dir + '/configuration.yaml', 'w') as f:
            f.write(yaml.dump(args))
    args['device'] = torch.device('cuda' if args['cuda'] else 'cpu')
    return args

if __name__ == '__main__':
    torch.set_num_threads(1)
    args = parse()
    set_context(args['num_units'], args['num_actors'])
    if global_dict['unit_idx'] == args['num_units']:
        evaluator(args)
    else:
        rank = global_dict['rank_local']
        if rank == global_dict['rank_learner']:
            learner(args)
        elif rank == global_dict['rank_replay']:
            replay(args)
        else:
            actor_id = rank
            actor(args, actor_id)
