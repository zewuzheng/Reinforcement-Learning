import torch
from torch import nn
from PPO_net import PPO_net
from torch.distributions import Beta
from Env_wrapper import Environ_test


import argparse
import torch
from PPO_training import PPO_train
from pyvirtualdisplay import Display

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('-gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('-img_stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('-seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('-render', action='store_true', default = False, help='render the environment')
parser.add_argument('-vis', action='store_true', default = False, help='use visdom')
parser.add_argument('-store', default='models/ppo_latest.pt', metavar= 'PATH',help='store the newest param to path')
parser.add_argument('-clip', type=float, default=0.1, metavar='G', help='set the PPO clip value epsilon')
parser.add_argument('-load', action='store_true', default = False, help='Load recent model')
parser.add_argument('-server', action='store_true', default = False, help='Fix display problem in server')
parser.add_argument('-actor_critic', action='store_true', default = False, help='Use actor critic agent')

args = parser.parse_args()
if args.server:
    virtual_display = Display(visible=0, size=(1, 1))
    virtual_display.start()

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("using device: ", device)

basic_config = {
    "ACTION_SIZE": (3,),
    "ACTOR_LR": 0.0001,
    "AC_STYLE": True,
    "BATCH_SIZE": 128,
    "BUFFER_SIZE": 2000,
    "CRITIC_LR": 0.0001,
    "DEVICE": device,
    "EPSILON": args.clip,
    "ENV_RENDER": args.render,
    "GAMMA": args.gamma,
    "GAME": "CarRacing-v0",
    "GAME_SEED": args.seed,
    "INIT_WEIGHT": True,
    "IMG_STACK": args.img_stack,
    "LR_RATE": 1e-3,
    "MAX_TRAIN_STEP": 100000,
    "PPO_EP": 10,
    "STATE_SIZE": (4, 96, 96),
    "STORE": args.store,
    "USE_VIS": args.vis,
    "LOAD_MODEL": args.load
}

from PPO_utils import Replay_buffer
# action = torch.randn((4, 3,))
# dist = Beta(2, 2)
# print(dist.log_prob(action).sum(dim = 1,keepdim = True))


#%%
## elementwise product
# a = torch.randn(3)
# b = torch.randn(3)
# print(a*b)


#%%
# ppo = PPO_net(basic_config)
# print(ppo.state_dict()["policy_mix.cnn_basic.0.bias"])
# ppo.load_model()
# print(ppo.state_dict()["policy_mix.cnn_basic.0.bias"])

#%%
# a = torch.randn(1, requires_grad= True)
# b = torch.randn(1, requires_grad= True)
# c = a*b
# d = torch.clamp(c, -0.5, 0.5)
# c.retain_grad()
# print("a",a.is_leaf)
# print("b",b.is_leaf)
# print("c",c.is_leaf)
# pint("d",d.is_leaf)
# print(d)
# d.backward)
# print(c.grad)

#%%
# alpha = torch.tensor([[1.5, 2, 2], [1, 2, 3]]).float().to(device)
# beta = torch.tensor([[2, 3, 4], [2,2,2]]).float().to(device)

# alpha = torch.tensor(2).float().to(device)
# beta = torch.tensor(3).float().to(device)
# m = Beta(alpha, beta)
# act = m.sample()
# if not act.size():
#     print("no action size")
# if act.dim() == 1 and act.size()[0] == 1:
#     print('single action')
#     act_logprob = m.log_prob(act)
# else:
#     print('multi-action')
#     act_logprob = m.log_prob(act).sum()
# print(act_logprob)

#%%
# import numpy as np
# import gym
# ppo = PPO_net(basic_config).double().to(device)
# env = Environ_test(basic_config)
# ppo.load_model('models/ppo_latest.pt')
# while True:
#     state = env.reset()
#     while True:
#         action, _ = ppo.get_action(state)
#         state_, reward, done, die = env.step(action* np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
#         env.render()
#         state = state_
#         if die:
#             break

#%%
# import gym
# import matplotlib.pyplot as plt
# import numpy as np
# env = gym.make('CarRacing-v0')
# state = env.reset()
# for _ in range(30):
#     action = [0.5,0.5,0]
#     state_,_,_,_ = env.step(action)
#
# gray = np.dot(state_[..., :], [0.299, 0.587, 0.114])
#
# plt.imshow(state_)
# plt.show()

#%%
import torch
import torch.nn as nn
a = PPO_net(basic_config)
a.load_model('models/ppo_latest_ac.pt')
print(a.state_dict())