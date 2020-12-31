import torch
from torch import nn
from PPO_net import PPO_net
from torch.distributions import Beta

cuda = torch.cuda.is_available()

device = torch.device('cuda' if cuda else 'cpu')
# device = 'cpu'
# print(torch.cuda.is_available())
basic_config = {
    "GAMMA": 0.9,
    "ACTOR_LR": 0.0001,
    "CRITIC_LR": 0.0001,
    "MIN_BATCH_SIZE": 64,
    "UPDATE_STEP": 15,
    "EPSILON": 0.2,
    "GAME": "CartPole-v0",
    "AC_STYLE": False,
    "INPUT_SIZE": 4,
    "INIT_WEIGHT": False,
    "STATE_SIZE": (4, 96, 96),
    "ACTION_SIZE": (3,),
    "PPO_EP": 10,
    "MAX_TRAIN_STEP": 100000,
    "DEVICE": device
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
import gym
env = gym.make("CartPole-v1")