"""
Module for DQN Model in Ape-X.
"""
import random
import torch
import torch.nn as nn
import numpy as np


def init_(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class DuelingDQN(nn.Module):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """
    def __init__(self, env, args):
        super(DuelingDQN, self).__init__()

        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.obs_high = np.mean(env.observation_space.high)
        self.obs_low = np.mean(env.observation_space.low)
        self.hidden_neurons = args['hidden_neurons']
        self.split = args['split_hidden_layer']

        self.features = self.feature_backbone()
        fc_neurons = self.hidden_neurons // 2 if self.split else self.hidden_neurons
        self.advantage = self.init(nn.Linear(fc_neurons, self.num_actions))
        self.value = self.init(nn.Linear(fc_neurons, 1))

    def init(self, module):
        return init_(module,
                     nn.init.orthogonal_,
                     lambda x: nn.init.constant_(x, 0),
                     nn.init.calculate_gain('relu'))

    def feature_backbone(self):
        if len(self.input_shape) == 3: # image input
            features = nn.Sequential(
                self.init(nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)),
                nn.ReLU(),
                self.init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
                nn.ReLU(),
                self.init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )
            feature_size = features(torch.zeros(1, *self.input_shape)).size(1)
            features = nn.Sequential(
                features,
                self.init(nn.Linear(feature_size, self.hidden_neurons)),
                nn.ReLU(),
            )
        elif len(self.input_shape) == 1: # vector input
            features = nn.Sequential(
                self.init(nn.Linear(self.input_shape[0], self.hidden_neurons)),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError(f'unknow input shape {self.input_shape}')
        return features

    def forward(self, x):
        x = (x - self.obs_low) / (self.obs_high - self.obs_low)
        x = self.features(x)
        if self.split:
            a, v = torch.split(x, self.hidden_neurons//2, 1)
        else:
            a, v = x, x
        advantage = self.advantage(a)
        value = self.value(v)
        return value + advantage - advantage.mean(1, keepdim=True)

    def act(self, state, epsilon):
        """
        Return action, max_q_value for given state
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.forward(state)

            if random.random() > epsilon:
                action = q_values.max(1)[1].item()
            else:
                action = random.randrange(self.num_actions)
        return action, q_values.numpy()[0]

class DDPG(nn.Module):
    def __init__(self, env, args):
        super(DDPG, self).__init__()

        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.obs_high = np.mean(env.observation_space.high)
        self.obs_low = np.mean(env.observation_space.low)

        self.state_features, self.action_features = self.feature_backbone()
        self.action_head = nn.Sequential(
            self.final_layer_init(nn.Linear(300, self.num_actions)),
            nn.Tanh(),
        )
        self.value_head = nn.Sequential(
            self.final_layer_init(nn.Linear(300, 1)),
        )
        self.actor_dict = nn.ModuleDict({
            'backbone': self.state_features,
            'head': self.action_head,
        })
        self.critic_dict = nn.ModuleDict({
            'backbone_state': self.state_features,
            'backbone_action': self.action_features,
            'head': self.value_head,
        })

    def forward(self, state, action=None):
        state = (state - self.obs_low) / (self.obs_high - self.obs_low)
        state = self.state_features(state)
        if action is not None:
            action = self.action_features(action)
            return self.value_head(state + action)
        else:
            return self.action_head(state)


    def final_layer_init(self, layer):
        return init_(
            layer,
            lambda x: nn.init.uniform_(x, -3e-3, 3e-3),
            lambda x: nn.init.uniform_(x, -3e-4, 3e-4),
        )
    def init(self, layer):
        fan_in = np.product(layer.weight.shape[1:])
        bound = 1 / np.sqrt(fan_in)
        return init_(
            layer,
            lambda x: nn.init.uniform_(x, -bound, bound),
            lambda x: nn.init.constant_(x, 0),
        )

    def feature_backbone(self):
        if len(self.input_shape) == 1: # vector input
            state_features = nn.Sequential(
                self.init(nn.Linear(self.input_shape[0], 400)),
                nn.ReLU(),
                self.init(nn.Linear(400, 300)),
                nn.ReLU()
            )
            action_features = nn.Sequential(
                self.init(nn.Linear(self.num_actions, 300)),
                nn.ReLU,
            )
        else:
            raise NotImplementedError(f'hasnot implement {self.input_shape}')
        return state_features, action_features


def build_model(env, args):
    return DuelingDQN(env, args)
