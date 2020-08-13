import random
import io
from PIL import Image

import numpy as np
import torch
import os
import logging
from datetime import datetime
import sys

global_dict = dict()

TAG_RECV_BATCH = 1
TAG_SEND_BATCH = 2
TAG_RECV_PRIOS = 3


def set_global_seeds(seed, use_torch=False):
    if use_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)

def compute_loss(model, tgt_model, batch, n_steps, gamma=0.99):
    states, actions, rewards, next_states, dones, weights = batch

    q_values = model(states)
    next_q_values = model(next_states)
    tgt_next_q_values = tgt_model(next_states)

    q_a_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_actions = next_q_values.max(1)[1].unsqueeze(1)
    next_q_a_values = tgt_next_q_values.gather(1, next_actions).squeeze(1)
    expected_q_a_values = rewards + (gamma ** n_steps) * next_q_a_values * (1 - dones)

    td_error = torch.abs(expected_q_a_values.detach() - q_a_values)
    prios = (td_error + 1e-6).data.cpu().numpy()

    loss = torch.where(td_error < 1, 0.5 * td_error ** 2, td_error - 0.5)
    loss = (loss * weights).mean()
    return loss, prios, q_values


